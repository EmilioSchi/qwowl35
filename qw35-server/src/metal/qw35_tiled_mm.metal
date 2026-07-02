// Qw35 tiled prompt matmul. This is the only retained DS4-derived kernel:
// Q4_K/Q5_K/Q6_K weight tiles multiplied by F32 prompt activations.

struct qw35_metal_args_mul_mm {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

constant bool QW35_MUL_MM_BOUNDS_INP [[function_constant(FC_MUL_MM + 0)]];
constant bool QW35_MUL_MM_BOUNDS_OUT [[function_constant(FC_MUL_MM + 1)]];

template<
    typename S0,
    typename S0_4x4,
    typename S0_8x8,
    typename S1,
    typename S1_2x4,
    typename S1_8x8,
    typename block_q,
    short nl,
    void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T1,
    typename T1_2x4>
kernel void qw35_kernel_mul_mm(
        constant qw35_metal_args_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im%args.ne12;
    const int i13 = im/args.ne12;
    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short offset1 = il0/nl;

    device const block_q * x =
        (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);
    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        S0_4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        if (QW35_MUL_MM_BOUNDS_INP) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;
                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                const short ib = 4*sx + sy;
                *(sb + 64*ib + 8*ly + lx) =
                    loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) =
                (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (!QW35_MUL_MM_BOUNDS_OUT || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C = (device float *) dst
            + (r0 + 32*(sgitg &  1))
            + (r1 + 16*(sgitg >> 1)) * args.ne0
            + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str =
            ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D =
                    (device float *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;
                threadgroup float * C = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// Activations (src1) stay f32 in threadgroup memory: staging them as f16
// costs ~1.4 logit drift per layer stack and matches neither the scalar
// decode matvec nor upstream llama.cpp, which also keeps f32 src1 as float.
typedef decltype(qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_q4_k, 16, qw35_dequantize_q4_k,
    float, float2x4>) qw35_mul_mm_t;

template [[host_name("qw35_mul_mm_q4_k_f32")]]
kernel qw35_mul_mm_t qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_q4_k, 16, qw35_dequantize_q4_k,
    float, float2x4>;

// Unified .qw35 stores the FFN as GF4; prefill dequantizes it through the same
// template (256-elem super-block, nl=16). qw35_block_gf4 / qw35_dequantize_gf4
// live in qw35_gf4.metal, which is concatenated before this file.
template [[host_name("qw35_mul_mm_gf4_f32")]]
kernel qw35_mul_mm_t qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_gf4, 16, qw35_dequantize_gf4,
    float, float2x4>;

// GF2 mirrors GF4's tiled path: interleaved 80-byte super-block (16 code
// words + 16 scale bytes per 256 elems), qw35_block_gf2 / qw35_dequantize_gf2
// in qw35_gf2.metal.
template [[host_name("qw35_mul_mm_gf2_f32")]]
kernel qw35_mul_mm_t qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_gf2, 16, qw35_dequantize_gf2,
    float, float2x4>;

template [[host_name("qw35_mul_mm_q5_k_f32")]]
kernel qw35_mul_mm_t qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_q5_k, 16, qw35_dequantize_q5_k,
    float, float2x4>;

template [[host_name("qw35_mul_mm_q6_k_f32")]]
kernel qw35_mul_mm_t qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_q6_k, 16, qw35_dequantize_q6_k,
    float, float2x4>;

template [[host_name("qw35_mul_mm_q8_0_f32")]]
kernel qw35_mul_mm_t qw35_kernel_mul_mm<
    half, half4x4, simdgroup_half8x8,
    float, float2x4, simdgroup_float8x8,
    qw35_block_q8_0, 2, qw35_dequantize_q8_0,
    float, float2x4>;
