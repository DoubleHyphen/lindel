//! Welcome to `lindel`, the **lin**earisation and **del**inearisation crate! In here you will find functions implementing encoding and decoding operations for Z-indexing and Hilbert indexing, which help linearise data-points while preserving locality.
//! 
//! # Usage
//! As far as primitive integers are concerned, this crate offers functions that can work either as methods (eg [`x.z_index()`](Lineariseable::z_index())) or as stand-alone functions (eg [`morton_encode(x)`](fn@morton_encode). All methods are defined within the [`Lineariseable`](trait@Lineariseable) trait.
//! 
//! The program essentially offers two 1-on-1 mappings between arrays and integers; if the array data-type is known, the integer type is selected automatically. For encoding operations, the array data-type is the input, which means that the compiler automatically knows which data-type the output should be. For decoding operations, however, the same integer can be decoded to arrays of different sizes; for that reason, the user will need to signify the output data-type somehow.
//! 
//! Please find illustrating examples below:
//! 
//! ### Z-indexing
//! ```
//! # use lindel::*;
//! let input = [5u8, 4, 12, 129]; // Input is known to be [u8; 4]
//! let z_index_1 = morton_encode(input); // Function style
//! let z_index_2 = input.z_index(); // Method style
//! assert_eq!(z_index_1, 268447241);
//! assert_eq!(z_index_1, z_index_2);
//! 
//! let reassembled_input_1: [u8; 4] = morton_decode(z_index_1);
//! // Output's data-type must be provided by the user
//! 
//! let reassembled_input_2 = <[u8; 4]>::from_z_index(z_index_2);
//! // Specifying the output data-type with method style is a little more involved.
//! 
//! assert_eq!(input, reassembled_input_1);
//! assert_eq!(input, reassembled_input_2);
//! ```
//! 
//! ### Hilbert indexing
//! ```
//! # use lindel::*;
//! let input = [0xDEADBEEFu32, 0xFACADE5]; // Input is known to be [u32; 2]
//! let hilbert_index_1 = hilbert_encode(input); // Function style
//! let hilbert_index_2 = input.hilbert_index(); // Method style
//! assert_eq!(hilbert_index_1, 17414049806762354884);
//! assert_eq!(hilbert_index_1, hilbert_index_2);
//! 
//! let reassembled_input_1: [u32; 2] = hilbert_decode(hilbert_index_1);
//! // Output's data-type must be provided by the user
//!
//! let reassembled_input_2 = <[u32; 2]>::from_hilbert_index(hilbert_index_2);
//! // Specifying the output data-type with method style is a little more involved.
//!
//! assert_eq!(input, reassembled_input_1);
//! assert_eq!(input, reassembled_input_2);
//! ```
//! 
//! # Morton encoding (Z-indexing)
//! This crate re-exports everything from the [`morton_encoding`](morton_encoding) crate for this operation; the user is cordially invited to look there for further information.
//! 
//! # Hilbert encoding
//! ## Algorithm details
//! The code for the Hilbert encoding is based on an algorithm by John Skilling, as implemented by Paul Chernoch. This algorithm has the disadvantage of only examining one bit at a time, but as a result it manages to avoid the very expensive computation of orientation that other algorithms have to perform. We don't know what the theoretical fastest is, if any; this algorithm was selected on an “eh, good enough” basis.
//! 
//! ## Implementation details
//! Our code is in essence a much-needed refactoring of mr Chernoch's implementation, clarifying the original source code by a lot, leveraging the type-system to help with correctness, and improving the efficiency of the code. To wit:
//! 1. Instead of accepting arbitrarily-sized slices as input, we accept arrays, to help ensure that each key comes from the encoding of the same amount of dimensions.
//! 2. Since the total amount of bits in the input is now known statically, our implementation outputs primitive integers instead of dynamically-sized `BigUint`s. This also allows it to work in `no_std` environments.
//! 3. Seeing as we've already implemented `morton_encoding`, we leverage it to perform what mr Chernoch calls “transposition” and what mr Skilling doesn't even deign to mention is necessary.
#![cfg_attr(
    feature = "nalg",
    doc = "4. Given that the [`nalgebra`](nalgebra) crate exists, we opted to just implement this crate's linearisation methods for `nalgebra`'s [`Point`](nalgebra::Point) data-types rather than re-implement them as mr Chernoch already did, so as to avail ourselves of `nalgebra`'s static correctness and sheer performance."
)]
//! 
//! 
//! ## Note about leading zeros
//! In contrast to Morton encoding, Hilbert encoding has the quirk of being dependent on the amount of leading zeros. As a result, because the amount of operations done by Skilling's algorithm is linerarly dependent on the amount of bits examined, it's imperative that one doesn't waste time examining useless bits. It is the solution to this problem that presents a difference between the code we found and the one that ended up in our implementation.
//! 
//! Messrs Skilling and Chernoch solved this problem by taking the amount of bits to be examined as a parametre to the function. This was crucially important to them, because their implementation only accepts `u32`s as its input, which would otherwise mean 32 bits to examine irrespective of the magnitude of the input.
//! 
//! Our solution, on the other hand, can accept coordinates as small as `u8`s. As a result, any data-set which is statically known to contain small enough numbers can simply be modelled with a smaller coordinate data-type, solving the biggest part of this problem in one fell swoop. Nonetheless, we also examine the leading zeros of our coordinates, and skip any leading zeros that come in groups of `D`, where `D` the amount of dimensions; this outputs the exact same results as one would get by examining all bits from the beginning, and allows us to avoid taking the amount of bits as a parametre. Nonetheless, we admittedly haven't benchmarked the cost of zero-counting to compare it to the other costs, because such micro-optimisations were deemed beyond the scope of this crate.
//! 
//! # Nalgebra
//! Hidden behind the `nalg` feature, so as to avoid dragging unnecessary dependencies to people who don't need them, is the 
#![cfg_attr(
    feature = "nalg",
    doc = "[`nalgebra_points`](nalgebra_points)"
)]
#![cfg_attr(
    not(feature = "nalg"),
    doc = "`nalgebra_points`"
)]
//! module, which offers all of those methods for `Point`s. 
#![cfg_attr(
    feature = "nalg",
    doc = "Please refer there for more information, although there isn't much to be said."
)]
#![cfg_attr(
    not(feature = "nalg"),
    doc = "However, the crate (or its documentation, in any event) has not currently been compiled with the `nalg` feature enabled, so it can't offer this functionality."
)]

//!
//! # Compact Hilbert encoding
//! Implemented by Chris Hamilton and copied with permission and gratitude into our own code. Information on implementation details and performance characteristics will have to wait until mr Hamilton can explain as much.
//!
//!
//!
//!
//!

/// If you can read this, it must mean that you're looking into the source code instead of reading the documentation. Thus, it needs to be acknowledged: _Yes_, I did basically copy-paste the functions from here to the `nalgebra_points` and `new_uints` modules. No, with how immature const generics are still, I'm not liable to change it any time soon. I do accept merge requests, however!

pub mod new_uints;
pub use morton_encoding;
pub use morton_encoding::*;
#[cfg(feature = "nalg")]
pub mod nalgebra_points;

use core::convert::From;
use core::ops::BitAndAssign;
use core::ops::BitOrAssign;
use core::ops::ShlAssign;
use num::traits::int::PrimInt;
use num_traits::ToPrimitive;
use num::Zero;

/// General trait for 
pub trait Lineariseable<Key> {
    fn z_index(&self) -> Key;
    fn hilbert_index(&self) -> Key;
    fn from_z_index(input: Key) -> Self;
    fn from_hilbert_index(input: Key) -> Self;
}

impl<N, const D: usize> Lineariseable<<N as IdealKey<D>>::Key> for [N; D]
where
    N: IdealKey<D>,
    N: ToPrimitive
        + Copy
        + PrimInt
        + BitOrAssign
        + BitAndAssign
        + std::ops::BitXorAssign,
    <N as IdealKey<D>>::Key:
        PrimInt 
        + From<N> 
        + BitOrAssign 
        + BitAndAssign 
        + ShlAssign<usize> 
        + std::ops::BitXorAssign,
{
/// A thin wrapper around the [`morton_encode`](fn@morton_encode) function.
    fn z_index(&self) -> <N as IdealKey<D>>::Key {
        morton_encode(*self)
    }
    
    fn hilbert_index(&self) -> <N as IdealKey<D>>::Key {
    
        let inverse_gray_encoding = |mut x| -> <N as IdealKey<D>>::Key {
            let log_bits: u32 = (std::mem::size_of::<<N as IdealKey<D>>::Key>() * 8)
                .next_power_of_two()
                .trailing_zeros();
            let powers_of_two = (0..log_bits).map(|i| 1<<i);
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
        min_leading_zeros /= D;
        min_leading_zeros *= D;
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

/// A thin wrapper around the [`morton_decode`](fn@morton_decode) function.
    fn from_z_index(input: <N as IdealKey<D>>::Key) -> Self {
        morton_decode(input)
    }
    
    fn from_hilbert_index(input: <N as IdealKey<D>>::Key) -> Self {
        let coor_bits = std::mem::size_of::<N>() * 8;
        let dims = D;
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

/// A thin wrapper around the [`from_hilbert_index`](Lineariseable::from_hilbert_index()) method.
pub fn hilbert_decode<Coordinate, const N: usize>(
    input: <Coordinate as IdealKey<N>>::Key,
) -> [Coordinate; N]
where
    Coordinate: IdealKey<N> 
    + ToPrimitive 
    + PrimInt
    + BitOrAssign
    + BitAndAssign
    + core::ops::BitXorAssign,
    <Coordinate as IdealKey<N>>::Key: ValidKey<Coordinate>
    + core::ops::BitXorAssign,
{
    <[Coordinate; N]>::from_hilbert_index(input)
}

/// A thin wrapper around the [`hilbert_index`](Lineariseable::hilbert_index()) method.
pub fn hilbert_encode<Coordinate, const N: usize>(
    input: [Coordinate; N],
) -> <Coordinate as IdealKey<N>>::Key
where
    Coordinate: IdealKey<N> 
    + ToPrimitive 
    + PrimInt
    + BitOrAssign
    + BitAndAssign
    + core::ops::BitXorAssign,
    <Coordinate as IdealKey<N>>::Key: ValidKey<Coordinate>
    + core::ops::BitXorAssign,
{
    input.hilbert_index()
}
