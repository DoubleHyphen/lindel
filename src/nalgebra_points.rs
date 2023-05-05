//! A few traits and functions for linearising `nalgebra`'s `Point` data-types.
//!
//! For people who've understood the functions in the main library, there's absolutely nothing new here. The only difference is that, rather than implementing each function in both method and function ways, each function is only implemented method-style (for the encoding) or associated-function style (for the decoding). Please refer to the examples mentioned in the [`Lineariseable`](Lineariseable) trait or the `README` file.
use morton_encoding::{bloat_custom_checked, nz, shrink_custom_checked};
use nalgebra::*;
use num::PrimInt;
use num_traits::Zero;
use std::ops::{BitAndAssign, BitOrAssign, ShlAssign};

/// Given a data type and a type-number, it yields the smallest unsigned
/// integer that's at least `N` times larger.
///
/// Implemented by brute force.

pub trait IdealKey<N> {
    type Key;
}

// UGLY WORK-AROUND, HOOOOOOOO!
impl IdealKey<U1> for u8 {
    type Key = Self;
}
impl IdealKey<U2> for u8 {
    type Key = Self;
}
impl IdealKey<U3> for u8 {
    type Key = Self;
}
impl IdealKey<U4> for u8 {
    type Key = Self;
}
impl IdealKey<U5> for u8 {
    type Key = Self;
}
impl IdealKey<U6> for u8 {
    type Key = Self;
}
impl IdealKey<U7> for u8 {
    type Key = Self;
}
impl IdealKey<U8> for u8 {
    type Key = Self;
}
impl IdealKey<U9> for u8 {
    type Key = Self;
}
impl IdealKey<U10> for u8 {
    type Key = Self;
}
impl IdealKey<U11> for u8 {
    type Key = Self;
}
impl IdealKey<U12> for u8 {
    type Key = Self;
}
impl IdealKey<U13> for u8 {
    type Key = Self;
}
impl IdealKey<U14> for u8 {
    type Key = Self;
}
impl IdealKey<U15> for u8 {
    type Key = Self;
}
impl IdealKey<U16> for u8 {
    type Key = Self;
}

impl IdealKey<U1> for u16 {
    type Key = Self;
}
impl IdealKey<U2> for u16 {
    type Key = Self;
}
impl IdealKey<U3> for u16 {
    type Key = Self;
}
impl IdealKey<U4> for u16 {
    type Key = Self;
}
impl IdealKey<U5> for u16 {
    type Key = Self;
}
impl IdealKey<U6> for u16 {
    type Key = Self;
}
impl IdealKey<U7> for u16 {
    type Key = Self;
}
impl IdealKey<U8> for u16 {
    type Key = Self;
}

impl IdealKey<U1> for u32 {
    type Key = Self;
}
impl IdealKey<U2> for u32 {
    type Key = Self;
}
impl IdealKey<U3> for u32 {
    type Key = Self;
}
impl IdealKey<U4> for u32 {
    type Key = Self;
}

impl IdealKey<U1> for u64 {
    type Key = Self;
}
impl IdealKey<U2> for u64 {
    type Key = Self;
}

impl IdealKey<U1> for u128 {
    type Key = Self;
}

use super::Lineariseable;

impl<N, D> Lineariseable<<N as IdealKey<D>>::Key> for Point<N, D>
where
    N: IdealKey<D>,
    D: DimName,
    N: Scalar
        + num::ToPrimitive
        + Copy
        + PrimInt
        + BitOrAssign
        + BitAndAssign
        + std::ops::BitXorAssign,
    <N as IdealKey<D>>::Key:
        PrimInt + From<N> + BitOrAssign + BitAndAssign + ShlAssign<usize> + std::ops::BitXorAssign,
    DefaultAllocator: nalgebra::allocator::Allocator<N, D>,
{
    /// The 'Point' analogue for the primitive types' [`z_index`](Lineariseable::z_index) method.
    /// # Examples
    /// ```rust
    /// use lindel::Lineariseable;
    /// let pnt = nalgebra::Point4::<u32>::new(51, 32, 65565, 10101010);
    /// assert_eq!(pnt.z_index(), 4953044972870758573802201754);
    /// ```
    fn z_index(&self) -> <N as IdealKey<D>>::Key {
        self.iter()
            .map(|&m| {
                bloat_custom_checked::<N, <N as IdealKey<D>>::Key>(m, nz(<D as DimName>::dim()))
                    .unwrap()
            })
            .fold(<N as IdealKey<D>>::Key::zero(), |acc, x| (acc << 1) | x)
    }

    /// The 'Point' analogue for the primitive types' [`hilbert_index`](Lineariseable::hilbert_index) method.
    /// # Examples
    /// ```rust
    /// use lindel::Lineariseable;
    /// let pnt = nalgebra::Point4::<u32>::new(51, 32, 65565, 10101010);
    /// assert_eq!(pnt.hilbert_index(), 4961111386034627444566496746);
    /// ```
    fn hilbert_index(&self) -> <N as IdealKey<D>>::Key {
        let inverse_gray_encoding = |mut x| -> <N as IdealKey<D>>::Key {
            let log_bits: u32 = (std::mem::size_of::<<N as IdealKey<D>>::Key>() * 8)
                .next_power_of_two()
                .trailing_zeros();
            let powers_of_two = (0..log_bits).map(|i| 1 << i);
            for pow in powers_of_two {
                x ^= x >> pow
            }
            x
        };

        let bits: usize = std::mem::size_of::<N>() * 8;
        let mut min_leading_zeros =
            self.iter().fold(N::zero(), |a, &b| a | b).leading_zeros() as usize;
        if min_leading_zeros == bits {
            return <N as IdealKey<D>>::Key::zero();
        }
        min_leading_zeros /= <D as DimName>::dim();
        min_leading_zeros *= <D as DimName>::dim();
        let bits = bits - min_leading_zeros;

        let mut input = self.clone();

        // Inverse undo
        for single_bit_mask in (1..bits).map(|x| N::one() << x).rev() {
            // We go from MSB to LSB

            let current_bit_is_set = |x| x & single_bit_mask != N::zero();

            let less_significant_bit_mask = single_bit_mask - N::one();

            // We do not need to XOR input[0] with t twice since they cancel each other out.
            let mut first_element = input[0];
            if current_bit_is_set(first_element) {
                first_element ^= less_significant_bit_mask; // invert
            }
            for x_i in input.iter_mut().skip(1) {
                if current_bit_is_set(*x_i) {
                    first_element ^= less_significant_bit_mask; // invert
                } else {
                    let t = (first_element ^ *x_i) & less_significant_bit_mask;
                    first_element ^= t;
                    *x_i ^= t;
                }
            }
            input[0] = first_element;
        } // exchange

        inverse_gray_encoding(input.z_index())
    }

    /// The 'Point' analogue for the primitive types' [`z_index`](Lineariseable::from_z_index) method.
    /// # Examples
    /// ```rust
    /// use lindel::Lineariseable;
    /// type Pnt = nalgebra::Point4::<u32>;
    /// let pnt = Pnt::new(51, 32, 65565, 10101010);
    /// let z_ind = pnt.z_index();
    /// let pnt_again = Pnt::from_z_index(z_ind);
    /// assert_eq!(pnt, pnt_again);
    /// ```
    fn from_z_index(input: <N as IdealKey<D>>::Key) -> Self {
        let size_ratio = <D as DimName>::dim();
        let mut result = Self::origin();
        for (i, element) in result.iter_mut().enumerate() {
            *element = shrink_custom_checked::<N, <N as IdealKey<D>>::Key>(
                input >> (size_ratio - i - 1),
                nz(<D as DimName>::dim()),
            )
            .unwrap()
        }
        result
    }

    /// The 'Point' analogue for the primitive types' [`hilbert_index`](Lineariseable::from_hilbert_index) method.
    /// # Examples
    /// ```rust
    /// use lindel::Lineariseable;
    /// type Pnt = nalgebra::Point4::<u32>;
    /// let pnt = Pnt::new(51, 32, 65565, 10101010);
    /// let h_ind = pnt.hilbert_index();
    /// let pnt_again = Pnt::from_hilbert_index(h_ind);
    /// assert_eq!(pnt, pnt_again);
    /// ```    
    fn from_hilbert_index(input: <N as IdealKey<D>>::Key) -> Self {
        let coor_bits = std::mem::size_of::<N>() * 8;
        let dims = <D as DimName>::dim();
        let mut min_leading_zeros = input.leading_zeros() as usize;

        let key_bits = std::mem::size_of::<<N as IdealKey<D>>::Key>() * 8;
        let useless_bits = key_bits - (dims * coor_bits);
        min_leading_zeros -= useless_bits;

        min_leading_zeros /= dims;
        min_leading_zeros /= dims;
        min_leading_zeros *= dims;
        let bits = coor_bits - min_leading_zeros;
        let gray_encoded_input = input ^ (input >> 1);
        let mut output = Self::from_z_index(gray_encoded_input);

        // Inverse undo
        for single_bit_mask in (1..bits).map(|x| N::one() << x) {
            // We go from LSB to MSB

            let current_bit_is_set = |x| x & single_bit_mask != N::zero();

            let less_significant_bit_mask = single_bit_mask - N::one();

            // We do not need to XOR input[0] with t twice since they cancel each other out.
            let mut first_element = output[0];
            for i in (1..dims).rev() {
                let x_i = &mut output[i];
                if current_bit_is_set(*x_i) {
                    first_element ^= less_significant_bit_mask; // invert
                } else {
                    let t = (first_element ^ *x_i) & less_significant_bit_mask;
                    first_element ^= t;
                    *x_i ^= t;
                }
            }
            if current_bit_is_set(first_element) {
                first_element ^= less_significant_bit_mask; // invert
            }
            output[0] = first_element;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(debug_assertions)]
    const TOTAL_BITS_USED: usize = 10;
    #[cfg(not(debug_assertions))]
    const TOTAL_BITS_USED: usize = 19;

    macro_rules! check_vals {
        ($coor: ty, $dims: ty) => {
            type Set = Point<$coor, $dims>;
            type Key = <$coor as IdealKey<$dims>>::Key;

            fn are_adjacent(x: Set, y: Set) -> bool {
                fn abs_diff((a, b): (&$coor, &$coor)) -> $coor {
                    if a > b {
                        a - b
                    } else {
                        b - a
                    }
                }
                x.iter().zip(y.iter()).map(abs_diff).sum::<$coor>() == 1
            }

            let coor_bits = std::mem::size_of::<$coor>() * 8;
            let dims = <$dims as DimName>::dim();
            let useful_bits = (dims * coor_bits) as u32;

            let big_limit = (1 as Key)
                .checked_shl(TOTAL_BITS_USED as u32)
                .unwrap_or(Key::MAX);

            use rand::Rng;
            let creation = Set::from_hilbert_index;
            let mut rng = rand::thread_rng();
            let limit = (1 as Key)
                .checked_shl(useful_bits)
                .map(|x| x - big_limit)
                .unwrap_or(1);
            let beginning = rng.gen_range(0..limit);
            let test_this = (0..big_limit)
                .map(|x| x + beginning)
                .map(|x| (creation(x), creation(x + 1)));

            for (a, b) in test_this {
                let thing_1 = a.z_index();
                let thing_2 = Set::from_z_index(thing_1);
                assert_eq!(a, thing_2);

                let thing_1 = a.hilbert_index();
                let thing_2 = Set::from_hilbert_index(thing_1);
                assert_eq!(a, thing_2);

                assert!(are_adjacent(a, b));
            }
        };
    }

    #[test]
    fn u8_2d() {
        check_vals!(u8, U2);
    }

    #[test]
    fn u8_3d() {
        check_vals!(u8, U3);
    }

    #[test]
    fn u8_4d() {
        check_vals!(u8, U4);
    }

    #[test]
    fn u8_5d() {
        check_vals!(u8, U5);
    }

    #[test]
    fn u8_6d() {
        check_vals!(u8, U6);
    }

    #[test]
    fn u8_7d() {
        check_vals!(u8, U7);
    }

    #[test]
    fn u8_8d() {
        check_vals!(u8, U8);
    }

    #[test]
    fn u16_2d() {
        check_vals!(u16, U2);
    }

    #[test]
    fn u16_3d() {
        check_vals!(u16, U3);
    }

    #[test]
    fn u16_4d() {
        check_vals!(u16, U4);
    }

    #[test]
    fn u32_3d() {
        check_vals!(u32, U3);
    }

    #[test]
    fn u32_4d() {
        check_vals!(u32, U4);
    }

    #[test]
    fn u64_2d() {
        check_vals!(u64, U2);
    }
}
