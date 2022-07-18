use concrete_commons::parameters::{CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweSize, MonomialDegree, PolynomialSize};
use concrete_core::backends::core::private as ccore;
use ccore::crypto::ggsw::StandardGgswCiphertext;
use concrete_core::backends::core::private::crypto::bootstrap::{FourierBuffers};
use concrete_core::backends::core::private::crypto::ggsw::FourierGgswCiphertext;
use concrete_core::backends::core::private::math::fft::Complex64;
use num_traits::identities::Zero;
use crate::rlwe::{RLWECiphertext, RLWEKeyswitchKey};
use crate::*;

#[derive(Debug, Clone)]
/// An RGSW ciphertext.
/// It is a wrapper around `StandardGgswCiphertext` from concrete.
pub struct RGSWCiphertext(pub(crate) StandardGgswCiphertext<Vec<Scalar>>);

impl RGSWCiphertext {
    pub fn allocate(poly_size: PolynomialSize, decomp_base_log: DecompositionBaseLog, decomp_level: DecompositionLevelCount) -> RGSWCiphertext {
        // TODO consider using Fourier version
        RGSWCiphertext(
            StandardGgswCiphertext::allocate(
                Scalar::zero(),
                poly_size,
                GlweSize(2),
                decomp_level,
                decomp_base_log,
            )
        )
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }

    pub fn ciphertext_count(&self) -> CiphertextCount {
        self.0.as_glwe_list().ciphertext_count()
    }

    pub(crate) fn external_product_with_buf_glwe<C>(&self, out: &mut GlweCiphertext<C>, d: &RLWECiphertext, buffers: &mut FourierBuffers<Scalar>)
        where C: AsMutSlice<Element=Scalar>
    {
        let mut transformed = FourierGgswCiphertext::allocate(
            Complex64::new(0., 0.),
            self.polynomial_size(),
            GlweSize(2),
            self.decomposition_level_count(),
            self.decomposition_base_log(),
        );
        transformed.fill_with_forward_fourier(&self.0, buffers);
        transformed.external_product(out, &d.0, buffers);
    }

    pub fn external_product_with_buf(&self, out: &mut RLWECiphertext, d: &RLWECiphertext, buffers: &mut FourierBuffers<Scalar>) {
        self.external_product_with_buf_glwe(&mut out.0, d, buffers);
    }

    pub fn external_product(&self, out: &mut RLWECiphertext, d: &RLWECiphertext) {
        let mut buffers = FourierBuffers::new(self.polynomial_size(), GlweSize(2));
        self.external_product_with_buf(out, d, &mut buffers);
    }

    pub fn cmux(&self, out: &mut RLWECiphertext, ct0: &RLWECiphertext, ct1: &RLWECiphertext) {
        let mut buffers = FourierBuffers::new(self.polynomial_size(), GlweSize(2));
        self.cmux_with_buf(out, ct0, ct1, &mut buffers);
    }

    pub fn cmux_with_buf(&self, out: &mut RLWECiphertext, ct0: &RLWECiphertext, ct1: &RLWECiphertext, buffers: &mut FourierBuffers<Scalar>) {
        assert_eq!(ct0.polynomial_size(), ct1.polynomial_size());
        // TODO: consider removing tmp
        let mut tmp = RLWECiphertext::allocate(ct1.polynomial_size());
        tmp.0
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(ct1.0.as_tensor().as_slice());
        out.0
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(ct0.0.as_tensor().as_slice());
        tmp.update_with_sub(ct0);
        self.external_product_with_buf(out, &tmp, buffers);
    }

    pub fn get_last_row(&self) -> RLWECiphertext {
        self.get_nth_row(self.decomposition_level_count().0 * 2 - 1)
    }

    pub fn get_nth_row(&self, n: usize) -> RLWECiphertext {
        let mut glwe_ct = GlweCiphertext::allocate(Scalar::zero(), self.polynomial_size(), GlweSize(2));
        glwe_ct.as_mut_tensor().fill_with_copy(self.0.as_glwe_list().ciphertext_iter().nth(n).unwrap().as_tensor());
        RLWECiphertext(glwe_ct)
    }

    /// Convert the RLWE key switching key to a RGSW ciphertext.
    /// Not recommended to use.
    pub fn from_keyswitch_key(ksk: &RLWEKeyswitchKey) -> RGSWCiphertext {
        let ell = ksk.decomposition_level_count();
        let base_log = ksk.decomposition_base_log();
        let mut rgsw = RGSWCiphertext::allocate(
            ksk.polynomial_size(),
            base_log,
            ell);
        let ks = ksk.get_keyswitch_key();
        assert_eq!(rgsw.0.as_glwe_list().ciphertext_count().0, 2 * ell.0);
        assert_eq!(ks.len(), ell.0);
        let mut i = 0usize;
        for mut ct in rgsw.0.as_mut_glwe_list().ciphertext_iter_mut() {
            let level = i / 2;
            if i % 2 == 0 {
                ct.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0).update_with_wrapping_sub(
                    &ks[level].get_mask().as_polynomial_list().get_polynomial(0));
                ct.get_mut_body().as_mut_polynomial().update_with_wrapping_sub(
                    &ks[level].get_body().as_polynomial());
            } else {
                let shift: usize = (Scalar::BITS as usize) - base_log.0 * (level + 1);
                ct.get_mut_body().as_mut_polynomial().get_mut_monomial(MonomialDegree(0)).set_coefficient(1 << shift);
            }
            i += 1;
        }
        rgsw
    }

    /// Execute the key switching operation with a buffer when self is a key switching key.
    pub fn keyswitch_ciphertext_with_buf(&self, after: &mut RLWECiphertext, before: &RLWECiphertext, buffers: &mut FourierBuffers<Scalar>) {
        self.external_product_with_buf(after, before, buffers);
    }

    /// Execute the key switching operation when self is a key switching key.
    pub fn keyswitch_ciphertext(&self, after: &mut RLWECiphertext, before: &RLWECiphertext) {
        self.external_product(after, before);
    }
}

#[cfg(test)]
mod test {
    use concrete_core::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
    use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    use num_traits::One;
    use crate::rlwe::{compute_noise_binary, compute_noise_ternary, RLWESecretKey};
    use super::*;

    #[test]
    fn test_external_product() {
        let mut ctx = Context::default();
        let orig_pt = ctx.gen_binary_pt();

        let sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);

        let mut rgsw_ct_0 = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        let mut rgsw_ct_1 = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        sk.encrypt_constant_rgsw(&mut rgsw_ct_0, &Plaintext(Scalar::zero()), &mut ctx);
        sk.encrypt_constant_rgsw(&mut rgsw_ct_1, &Plaintext(Scalar::one()), &mut ctx);

        let mut rlwe_ct = RLWECiphertext::allocate(ctx.poly_size);
        sk.binary_encrypt_rlwe(&mut rlwe_ct, &orig_pt, &mut ctx);
        println!("initial noise: {:?}", compute_noise_binary(&sk, &rlwe_ct, &orig_pt));

        let zero_ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

        {
            let mut out_0 = RLWECiphertext::allocate(ctx.poly_size);
            let mut out_1 = RLWECiphertext::allocate(ctx.poly_size);
            rgsw_ct_0.external_product(&mut out_0, &rlwe_ct);
            rgsw_ct_1.external_product(&mut out_1, &rlwe_ct);

            let mut decrypted = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            sk.binary_decrypt_rlwe(&mut decrypted, &out_0);
            assert_eq!(decrypted, zero_ptxt);
            sk.binary_decrypt_rlwe(&mut decrypted, &out_1);
            assert_eq!(decrypted, orig_pt);
            println!("final noise: {:?}", compute_noise_binary(&sk, &out_1, &orig_pt));
        }
    }

    #[test]
    fn test_rgsw_shape() {
        let mut ctx = Context::default();
        ctx.poly_size = PolynomialSize(8);
        let sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);

        let mut rgsw_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);
        sk.0.trivial_encrypt_constant_ggsw(&mut rgsw_ct.0, &Plaintext(Scalar::one()), ctx.std, &mut ctx.encryption_generator);

        // the way RGSW ciphertext arranged is different from some literature
        // usually it's Z + m*G, where Z is the RLWE encryption of zeros and G is the gadget matrix
        //      g_1 0
        // G =  g_2 0
        //      0   g_1
        //      0   g_2
        // but concrete has it arragned like this:
        //      g_1 0
        // G =  0   g_1
        //      g_2 0
        //      0   g_2
        let mut level_count = 0;
        for m in rgsw_ct.0.level_matrix_iter() {
            let mut row_count = 0;
            for row in m.row_iter() {
                let ct = row.into_glwe();
                println!("mask : {:?}", ct.get_mask().as_polynomial_list().get_polynomial(0));
                println!("body: {:?}", ct.get_body().as_polynomial());
                row_count += 1;
            }
            assert_eq!(row_count, 2);
            level_count += 1;
        }
        assert_eq!(level_count, ctx.level_count.0);
    }

    #[test]
    fn test_keyswitching() {
        let mut ctx = Context::default();
        let messages = ctx.gen_ternary_ptxt();

        let sk_after = ctx.gen_rlwe_sk();
        let sk_before = ctx.gen_rlwe_sk();

        let mut ct_after = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_before = RLWECiphertext::allocate(ctx.poly_size);

        let mut ksk_slow = RLWEKeyswitchKey::allocate(
            ctx.ks_base_log,
            ctx.ks_level_count,
            ctx.poly_size,
        );
        ksk_slow.fill_with_keyswitch_key(&sk_before, &sk_after, ctx.std, &mut ctx.encryption_generator);
        let ksk = RGSWCiphertext::from_keyswitch_key(&ksk_slow);

        // encrypts with the before key our messages
        sk_before.ternary_encrypt_rlwe(&mut ct_before, &messages, &mut ctx);
        println!("msg before: {:?}", messages.as_tensor());
        let mut dec_messages_1 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_before.ternary_decrypt_rlwe(&mut dec_messages_1, &ct_before);
        println!("msg after dec: {:?}", dec_messages_1.as_tensor());
        println!("initial noise: {:?}", compute_noise_ternary(&sk_before, &ct_before, &messages));

        ksk.keyswitch_ciphertext(&mut ct_after, &ct_before);

        let mut dec_messages_2 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        sk_after.ternary_decrypt_rlwe(&mut dec_messages_2, &ct_after);
        println!("msg after ks: {:?}", dec_messages_2.as_tensor());

        assert_eq!(dec_messages_1, dec_messages_2);
        assert_eq!(dec_messages_1, messages);
        println!("final noise: {:?}", compute_noise_ternary(&sk_after, &ct_after, &messages));
    }

    #[test]
    fn test_cmux() {
        let mut ctx = Context::default();
        let pt_0 = ctx.gen_ternary_ptxt();
        let pt_1 = ctx.gen_ternary_ptxt();
        let sk = ctx.gen_rlwe_sk();

        let mut ct_0 = RLWECiphertext::allocate(ctx.poly_size);
        let mut ct_1 = RLWECiphertext::allocate(ctx.poly_size);
        sk.ternary_encrypt_rlwe(&mut ct_0, &pt_0, &mut ctx);
        sk.ternary_encrypt_rlwe(&mut ct_1, &pt_1, &mut ctx);

        let mut ct_gsw = RGSWCiphertext::allocate(ctx.poly_size, ctx.base_log, ctx.level_count);

        {
            // set choice bit to 0
            sk.encrypt_constant_rgsw(&mut ct_gsw, &Plaintext(Scalar::zero()), &mut ctx);
            let mut ct_result = RLWECiphertext::allocate(ctx.poly_size);
            let mut pt_result = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

            ct_gsw.cmux(&mut ct_result, &ct_0, &ct_1);
            sk.ternary_decrypt_rlwe(&mut pt_result, &ct_result);
            assert_eq!(pt_result, pt_0);
        }

        {
            // set choice bit to 1
            sk.encrypt_constant_rgsw(&mut ct_gsw, &Plaintext(Scalar::one()), &mut ctx);
            let mut ct_result = RLWECiphertext::allocate(ctx.poly_size);
            let mut pt_result = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

            ct_gsw.cmux(&mut ct_result, &ct_0, &ct_1);
            sk.ternary_decrypt_rlwe(&mut pt_result, &ct_result);
            assert_eq!(pt_result, pt_1);
        }
    }
}
