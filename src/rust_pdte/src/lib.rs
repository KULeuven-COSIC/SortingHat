#![allow(deprecated)]

pub mod rlwe;
pub mod rgsw;
pub mod decision_tree;

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;
use concrete_core::backends::core::private as ccore;
use ccore::math::polynomial::Polynomial;
use ccore::math::random::RandomGenerator;
use ccore::math::tensor::{AsMutSlice, Tensor, AsRefSlice, AsMutTensor, AsRefTensor};
use ccore::crypto::encoding::PlaintextList;
use ccore::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
use ccore::crypto::glwe::GlweCiphertext;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, GlweSize, MonomialDegree, PlaintextCount, PolynomialSize};
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev};
use concrete_core::backends::core::private::crypto::bootstrap::FourierBuffers;
use num_traits::{One, Zero};
use crate::rgsw::RGSWCiphertext;
use crate::rlwe::*;

pub type Scalar = u64;
pub type SignedScalar = i64;

/// The context structure holds the TFHE parameters and
/// random number generators.
pub struct Context {
    pub random_generator: RandomGenerator,
    pub secret_generator: SecretRandomGenerator,
    pub encryption_generator: EncryptionRandomGenerator,
    pub std: LogStandardDev,
    pub poly_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level_count: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level_count: DecompositionLevelCount,
    pub negs_base_log: DecompositionBaseLog,
    pub negs_level_count: DecompositionLevelCount,
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "q={:?}, poly_size={:?}, log_std={:?}, default_decomp=({:?},{:?}), ks_decomp=({:?},{:?}), negs_decomp=({:?},{:?})",
                 Scalar::BITS, self.poly_size.0, self.std.get_log_standard_dev(),
                 self.base_log.0, self.level_count.0, self.ks_base_log.0, self.ks_level_count.0, self.negs_base_log.0, self.negs_level_count.0)
    }
}

impl Context {
    /// Create the default context that is suitable for
    /// all experiments in the repository.
    pub fn default() -> Context {
        let random_generator = RandomGenerator::new(None);
        let secret_generator = SecretRandomGenerator::new(None);
        let encryption_generator = EncryptionRandomGenerator::new(None);
        let std = LogStandardDev::from_log_standard_dev(-55.);
        let poly_size = PolynomialSize(2048);
        let base_log = DecompositionBaseLog(4);
        let level_count = DecompositionLevelCount(7);
        let ks_base_log = DecompositionBaseLog(4);
        let ks_level_count = DecompositionLevelCount(8);
        let negs_base_log = DecompositionBaseLog(4);
        let negs_level_count = DecompositionLevelCount(8);
        Context {
            random_generator,
            secret_generator,
            encryption_generator,
            std,
            poly_size,
            base_log,
            level_count,
            ks_base_log,
            ks_level_count,
            negs_base_log,
            negs_level_count
        }
    }

    /// Output the plaintext count.
    pub fn plaintext_count(&self) -> PlaintextCount {
        PlaintextCount(self.poly_size.0)
    }

    /// Generate a binary plaintext.
    pub fn gen_binary_pt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator.fill_tensor_with_random_uniform_binary(ptxt.as_mut_tensor());
        ptxt
    }

    /// Generate a ternay plaintext.
    pub fn gen_ternary_ptxt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator.fill_tensor_with_random_uniform_ternary(ptxt.as_mut_tensor());
        ptxt
    }

    /// Generate a unit plaintext (all coefficients are 0 except the constant term is 1).
    pub fn gen_unit_pt(&self) -> PlaintextList<Vec<Scalar>> {
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), self.plaintext_count());
        *ptxt.as_mut_polynomial().get_mut_monomial(MonomialDegree(0)).get_mut_coefficient() = Scalar::one();
        ptxt
    }

    /// Generate a plaintext where all the coefficients are 0.
    pub fn gen_zero_pt(&self) -> PlaintextList<Vec<Scalar>> {
        PlaintextList::allocate(Scalar::zero(), self.plaintext_count())
    }

    /// Generate a RLWE secret key.
    pub fn gen_rlwe_sk(&mut self) -> RLWESecretKey {
        RLWESecretKey::generate_binary(self.poly_size, &mut self.secret_generator)
    }

    /// Allocate and return buffers that are used for FFT.
    pub fn gen_fourier_buffers(&self) -> FourierBuffers<Scalar> {
        FourierBuffers::new(self.poly_size, GlweSize(2))
    }
}

pub(crate) fn mul_const<C>(poly: &mut Tensor<C>, c: Scalar)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in poly.iter_mut() {
        *coeff = coeff.wrapping_mul(c);
    }
}

#[inline]
pub(crate) const fn log2(input: usize) -> usize {
    core::mem::size_of::<usize>() * 8 - (input.leading_zeros() as usize) - 1
}

/// Encode binary x as x*(q/2)
pub fn binary_encode(x: &mut Scalar) {
    assert!(*x == 0 || *x == 1);
    *x = *x << (Scalar::BITS - 1)
}

pub fn binary_decode(x: &mut Scalar) {
    let lower = Scalar::MAX as Scalar >> 2;
    let upper = lower + (Scalar::MAX as Scalar >> 1);
    if *x >= lower && *x < upper {
        *x = 1;
    } else {
        *x = 0;
    }
}

/// Encode ternary x as x*(q/3)
pub fn ternary_encode(x: &mut Scalar) {
    const THIRD: Scalar = (Scalar::MAX as f64 / 3.0) as Scalar;
    if *x == 0 {
        *x = 0;
    } else if *x == 1 {
        *x = THIRD;
    } else if *x == Scalar::MAX {
        *x = 2*THIRD;
    } else {
        panic!("not a ternary scalar")
    }
}

pub fn ternary_decode(x: &mut Scalar) {
    const SIXTH: Scalar = (Scalar::MAX as f64 / 6.0) as Scalar;
    const THIRD: Scalar = SIXTH + SIXTH;
    const HALF: Scalar = Scalar::MAX / 2;
    if *x > SIXTH && *x <= HALF {
        *x = 1;
    } else if *x > HALF && *x <= HALF + THIRD {
        *x = Scalar::MAX;
    } else {
        *x = 0;
    }
}

/// Encode a binary polynomial.
pub fn poly_binary_encode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        binary_encode(coeff);
    }
}

pub fn poly_binary_decode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        binary_decode(coeff);
    }
}

/// Encode a ternary polynomial.
pub fn poly_ternary_encode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        ternary_encode(coeff);
    }
}

pub fn poly_ternary_decode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        ternary_decode(coeff);
    }
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k<C>(poly: &Polynomial<C>, k: usize) -> Polynomial<Vec<Scalar>>
    where C: AsRefSlice<Element=Scalar>
{
    let mut out = Polynomial::allocate(Scalar::zero(), poly.polynomial_size());
    eval_x_k_in_memory(&mut out, poly, k);
    out
}

/// Evaluate f(x) on x^k, where k is odd
pub(crate) fn eval_x_k_in_memory<C>(out: &mut Polynomial<Vec<Scalar>>, poly: &Polynomial<C>, k: usize)
    where C: AsRefSlice<Element=Scalar>
{
    assert_eq!(k % 2, 1);
    assert!(poly.polynomial_size().0.is_power_of_two());
    *out.as_mut_tensor().get_element_mut(0) = *poly.as_tensor().get_element(0);
    for i in 1..poly.polynomial_size().0 {
        // i-th term becomes ik-th term, but reduced by n
        let j = i * k % poly.polynomial_size().0;
        let sign = if ((i * k) / poly.polynomial_size().0) % 2 == 0
        { 1 } else { Scalar::MAX };
        let c = *poly.as_tensor().get_element(i);
        *out.as_mut_tensor().get_element_mut(j) = sign.wrapping_mul(c);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_binary_encoder() {
        {
            let mut x: Scalar = 0;
            binary_encode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: Scalar = 1;
            binary_encode(&mut x);
            assert_eq!(x, 1 << (Scalar::BITS - 1));
        }
        {
            let mut x: Scalar = 10;
            binary_decode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: Scalar = Scalar::MAX;
            binary_decode(&mut x);
            assert_eq!(x, 0);
        }
        {
            let mut x: Scalar = 1 << (Scalar::BITS - 1);
            binary_decode(&mut x);
            assert_eq!(x, 1);
        }
    }
}

