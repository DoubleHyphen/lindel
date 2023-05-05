//! A macro that creates a new `[usize; N]` data-type that behaves as an integer for the purposes of accepting a new `Key`.
//!
//! The `create_lineariseable_data_type` macro accepts three arguments:
//! 1. The coordinate data-type, an unsigned integer
//! 2. The amount of dimensions that the data-set must have
//! 3. A name for the new data-type that will function as the key. (Must be provided by the user.)
//!
//! We haven't yet tried whether it can accept the new data-types it output as input coordinates; `u128`s were considered enough as a limit.
//!
//! # Example usage:
//! ```rust
//! lindel::create_lineariseable_data_type!(u128, 33, NewKey);
//! let input = [870u128; 33];
//! let hind = NewKey::hilbert_index(input);
//! let zind = NewKey::z_index(input);
//! let reinstated_input = hind.from_hilbert_index();
//! assert_eq!(input, reinstated_input);
//! let reinstated_input = NewKey::from_z_index(zind);
//! assert_eq!(input, reinstated_input);
//! ```
//!
//! # Note
//! Please bear in mind that the new data-type has only implemented whichever methods and operators were strictly necessary for its operation as a Key; for instance, only wrapping addition is implemented, and subtraction is lacking entirely.
#[macro_export]
macro_rules! create_lineariseable_data_type {
    ($coor: ty, $dim: expr, $key: ident) => {
        #[derive(Debug, Clone, Copy, Eq, PartialEq)]
        pub struct $key(pub [usize; Self::ARRAY_LENGTH]);

        impl core::cmp::Ord for $key {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.0.iter().rev().cmp(other.0.iter().rev())
            }
        }

        impl core::cmp::PartialOrd for $key {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl core::ops::AddAssign for $key {
            fn add_assign(&mut self, other: Self) {
                let mut carry: usize = 0;
                for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
                    let (result, other_carry) = (*a).overflowing_add(carry);
                    let (result, ooc) = result.overflowing_add(b);
                    carry = other_carry as usize + ooc as usize;
                    *a = result;
                }
            }
        }

        impl core::ops::Add for $key {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                let mut copy = self;
                copy += other;
                copy
            }
        }

        impl core::ops::BitOr for $key {
            type Output = Self;

            fn bitor(self, other: Self) -> Self::Output {
                let mut copy = self;
                copy |= other;
                copy
            }
        }

        impl core::ops::BitAnd for $key {
            type Output = Self;

            fn bitand(self, other: Self) -> Self::Output {
                let mut copy = self;
                copy &= other;
                copy
            }
        }

        impl core::ops::BitXor for $key {
            type Output = Self;

            fn bitxor(self, other: Self) -> Self::Output {
                let mut copy = self;
                copy ^= other;
                copy
            }
        }

        impl From<usize> for $key {
            fn from(x: usize) -> Self {
                let mut result = [0usize; Self::ARRAY_LENGTH];
                result[0] = x;
                Self(result)
            }
        }

        impl From<$coor> for $key {
            fn from(x: $coor) -> Self {
                use core::convert::TryFrom;
                use core::convert::TryInto;
                if Self::COOR_BYTES <= Self::USIZE_BYTES {
                    (x as usize).into()
                } else {
                    let mut result: Self = 0usize.into();
                    let mut input = x;
                    let mask = 0usize.wrapping_sub(1);
                    if let Ok(mask) = <$coor>::try_from(mask) {
                        for x in result.0.iter_mut() {
                            *x = (input & mask).try_into().unwrap();
                            input = input.wrapping_shr(64);
                        }
                        result
                    } else {
                        (x as usize).into()
                    }
                }
            }
        }

        impl core::convert::TryFrom<$key> for $coor {
            type Error = ();

            fn try_from(x: $key) -> core::result::Result<$coor, ()> {
                use core::convert::TryInto;
                if $key::COOR_BYTES <= $key::USIZE_BYTES {
                    let mask: usize = <$coor>::max_value().try_into().unwrap();
                    Ok((x.0[0] & mask).try_into().unwrap())
                } else {
                    let mut result: $coor = 0usize.try_into().unwrap();
                    let amt_of_usizes_in_a_coor =
                        ($key::COOR_BYTES + $key::USIZE_BYTES - 1) / $key::USIZE_BYTES;
                    for &a in x.0.iter().take(amt_of_usizes_in_a_coor).rev() {
                        result = result.wrapping_shl($key::USIZE_BITS.try_into().unwrap());
                        let a: $coor = a.try_into().unwrap();
                        result |= a;
                    }
                    Ok(result)
                }
            }
        }

        impl core::ops::BitAndAssign for $key {
            fn bitand_assign(&mut self, other: Self) {
                self.0
                    .iter_mut()
                    .zip(other.0.iter())
                    .for_each(|(a, &b)| *a &= b);
            }
        }

        impl core::ops::BitXorAssign for $key {
            fn bitxor_assign(&mut self, other: Self) {
                self.0
                    .iter_mut()
                    .zip(other.0.iter())
                    .for_each(|(a, &b)| *a ^= b);
            }
        }

        impl core::ops::BitOrAssign for $key {
            fn bitor_assign(&mut self, other: Self) {
                self.0
                    .iter_mut()
                    .zip(other.0.iter())
                    .for_each(|(a, &b)| *a |= b);
            }
        }

        impl core::ops::ShlAssign<usize> for $key {
            fn shl_assign(&mut self, x: usize) {
                let log2_of_usize_bits =
                    usize::MAX.count_ones().next_power_of_two().trailing_zeros();
                let full_words = x.wrapping_shr(log2_of_usize_bits);
                let remainder = x & (Self::USIZE_BITS - 1);
                for i in (full_words..self.0.len()).rev() {
                    self.0[i] = self.0[i - full_words];
                }
                self.0.iter_mut().take(full_words).for_each(|y| *y = 0);

                let remainder = remainder as u32;
                let siiigh = Self::USIZE_BITS as u32;

                if remainder != 0 {
                    for i in (1..self.0.len()).rev() {
                        self.0[i] = self.0[i].wrapping_shl(remainder);
                        self.0[i] |= (self.0[i - 1]).wrapping_shr(siiigh.wrapping_sub(remainder));
                    }
                    self.0[0] = self.0[0].wrapping_shl(remainder);
                }
            }
        }

        impl core::ops::ShrAssign<usize> for $key {
            fn shr_assign(&mut self, x: usize) {
                let log2_of_usize_bits =
                    usize::MAX.count_ones().next_power_of_two().trailing_zeros();
                let full_words = x.wrapping_shr(log2_of_usize_bits);
                let remainder = x & (Self::USIZE_BITS - 1);
                for i in full_words..self.0.len() {
                    self.0[i - full_words] = self.0[i];
                }
                self.0
                    .iter_mut()
                    .rev()
                    .take(full_words)
                    .for_each(|y| *y = 0);

                let remainder = remainder as u32;
                let siiigh = Self::USIZE_BITS as u32;

                if remainder != 0 {
                    for i in 1..self.0.len() {
                        let ii = i - 1;
                        self.0[ii] = self.0[ii].wrapping_shr(remainder);
                        self.0[ii] |= (self.0[i]).wrapping_shl(siiigh.wrapping_sub(remainder));
                    }
                    self.0[self.0.len() - 1] = self.0[self.0.len() - 1].wrapping_shr(remainder);
                }
            }
        }

        impl core::ops::Shl<usize> for $key {
            type Output = Self;

            fn shl(self, x: usize) -> Self::Output {
                let mut copy = self;
                copy <<= x;
                copy
            }
        }

        impl core::ops::Shr<usize> for $key {
            type Output = Self;

            fn shr(self, x: usize) -> Self::Output {
                let mut copy = self;
                copy >>= x;
                copy
            }
        }

        impl $key {
            const USIZE_BYTES: usize = core::mem::size_of::<usize>();
            const USIZE_BITS: usize = Self::USIZE_BYTES * 8;
            const COOR_BYTES: usize = core::mem::size_of::<$coor>();
            const COOR_BITS: usize = Self::COOR_BYTES * 8;
            const KEY_BYTES: usize = core::mem::size_of::<$key>();
            const KEY_BITS: usize = Self::KEY_BYTES * 8;
            const ARRAY_LENGTH: usize =
                ((Self::COOR_BYTES * $dim) + Self::USIZE_BYTES - 1) / Self::USIZE_BYTES;
            const MAX: Self = Self([0usize.wrapping_sub(1); Self::ARRAY_LENGTH]);

            fn pattern(amt_of_ones: usize) -> $key {
                let mut result: Self = 0usize.into();
                let log2_of_usize_bits =
                    usize::MAX.count_ones().next_power_of_two().trailing_zeros();
                let whole_words = amt_of_ones.wrapping_shr(log2_of_usize_bits);
                result
                    .0
                    .iter_mut()
                    .take(whole_words)
                    .for_each(|a| *a = 0usize.wrapping_sub(1));
                let remainder = amt_of_ones & (Self::USIZE_BITS - 1);
                result.0[whole_words] = (1 << remainder) - 1;
                result
            }

            fn get_mask(step_number: usize) -> $key {
                type Key = $key;
                let siz_rat = $dim;
                let tentative_mask = {
                    let key_bits = Self::KEY_BITS;
                    let amount_of_1s_per_pattern = 1 << step_number;
                    let pattern_length = siz_rat * amount_of_1s_per_pattern;
                    let pattern = Self::pattern(amount_of_1s_per_pattern);
                    let mut insert_patterns_here: Key = pattern;
                    let mut amt_of_patterns_we_need_to_insert = key_bits / pattern_length;
                    while amt_of_patterns_we_need_to_insert > 1 {
                        amt_of_patterns_we_need_to_insert -= 1;
                        insert_patterns_here <<= pattern_length;
                        insert_patterns_here |= pattern;
                    }
                    insert_patterns_here
                };

                let masking_is_necessary = {
                    use core::convert::TryInto;
                    let usize_bits: u32 = Self::USIZE_BITS.try_into().unwrap();
                    let floor_of_log2_of_siz_rat = usize_bits - siz_rat.leading_zeros() - 1;
                    ((step_number as u32) % floor_of_log2_of_siz_rat) == 0
                };
                if masking_is_necessary {
                    tentative_mask
                } else {
                    Key::MAX
                }
            }

            pub fn bloat(input: $coor) -> $key {
                let size_ratio = $dim as usize;
                let mut blot: $key = input.into();
                let mut shift_bitor_mask = |x| {
                    blot |= blot << ((size_ratio - 1) << x);
                    blot &= Self::get_mask(x);
                };
                let steps = Self::COOR_BITS.next_power_of_two().trailing_zeros() as usize;
                for x in (0usize..steps).rev() {
                    shift_bitor_mask(x);
                }
                blot
            }

            pub fn shrink(input: $key) -> $coor {
                use core::convert::TryInto;
                let size_ratio = $dim as usize;
                let mut rez: $key = input;
                let mut shift_bitor_mask = |x| {
                    rez &= Self::get_mask(x);
                    rez |= rez >> ((size_ratio - 1) << x);
                };
                let steps = Self::COOR_BITS.next_power_of_two().trailing_zeros() as usize;
                for x in 0usize..steps {
                    shift_bitor_mask(x);
                }
                let mask: $coor = 0;
                let mask = mask.wrapping_sub(1);
                let mask: $key = mask.into();
                (rez & mask).try_into().unwrap()
            }

            pub fn z_index(input: [$coor; $dim]) -> $key {
                let bloat_fn = |&x| Self::bloat(x);
                let zero: $key = 0usize.into();
                input
                    .iter()
                    .map(bloat_fn)
                    .fold(zero, |acc, x| (acc << 1) | x)
            }

            fn gray_code(input: $key) -> $key {
                input ^ (input >> 1)
            }

            fn from_gray_code(input: $key) -> $key {
                let limit = Self::KEY_BITS.next_power_of_two().trailing_zeros();
                let mut result = input;
                for i in (0..limit).map(|x| 1 << x) {
                    result ^= result >> i;
                }
                result
            }

            pub fn hilbert_index(input: [$coor; $dim]) -> $key {
                let mut bits: usize = Self::COOR_BITS;
                let mut min_leading_zeros =
                    input.iter().fold(0, |a, &b| a | b).leading_zeros() as usize;
                if min_leading_zeros == bits {
                    return 0usize.into();
                } else if min_leading_zeros >= $dim {
                    min_leading_zeros /= $dim;
                    min_leading_zeros *= $dim;
                    bits -= min_leading_zeros;
                }

                let mut input = input;

                // Inverse undo
                for single_bit_mask in (1..bits).map(|x| 1 << x).rev() {
                    // We go from MSB to LSB

                    let current_bit_is_set = |x| x & single_bit_mask != 0;

                    let less_significant_bit_mask = single_bit_mask - 1;

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

                $key::from_gray_code($key::z_index(input))
            }

            pub fn from_z_index(self) -> [$coor; $dim] {
                let size_ratio = $dim;
                let mut result: [$coor; $dim] = [0; $dim];
                for (i, element) in result.iter_mut().enumerate() {
                    *element = Self::shrink(self >> (size_ratio - i - 1));
                }
                result
            }

            pub fn from_hilbert_index(self) -> [$coor; $dim] {
                let bits: usize = Self::COOR_BITS;
                let dims = $dim;
                let input = Self::gray_code(self);
                let mut output = Self::from_z_index(input);

                let coor_one: $coor = 1u8.into();

                // Inverse undo
                for single_bit_mask in (1..bits).map(|i| coor_one << i) {
                    // We go from LSB to MSB

                    let current_bit_is_set = |x: $coor| x & single_bit_mask != 0;

                    let less_significant_bit_mask: $coor = single_bit_mask - coor_one;

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

            fn _z_index_naive(input: [$coor; $dim]) -> $key {
                let mut result: $key = 0usize.into();
                let mut key_mask: $key = 1usize.into();
                let coor_bits = Self::COOR_BITS;
                for coordinate_mask in (0..coor_bits).map(|x| 1 << x) {
                    for t in input.iter().rev() {
                        if t & coordinate_mask != 0 {
                            result |= key_mask;
                        }
                        key_mask <<= 1;
                    }
                }
                result
            }
        }
    };
}

#[cfg(test)]
mod tests {
    #[cfg(debug_assertions)]
    const TOTAL_BITS_USED: usize = 10;
    #[cfg(not(debug_assertions))]
    const TOTAL_BITS_USED: usize = 19;

    macro_rules! check_vals {
        ($coor: ty, $dims: expr) => {
            create_lineariseable_data_type!($coor, $dims, Keyyy);
            type Set = [$coor; $dims];

            fn get_random_key() -> Keyyy {
                let mut result: Keyyy = 0usize.into();
                result
                    .0
                    .iter_mut()
                    .for_each(|x| *x = rand::random::<usize>());
                let remaining_bytes = ($dims * Keyyy::COOR_BYTES) % Keyyy::USIZE_BYTES;
                let mask: usize = 1 << (remaining_bytes * 8);
                let mask = mask - 1;
                result.0.iter_mut().rev().take(1).for_each(|a| (*a) &= mask);
                result
            }

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

            let big_limit = 1usize << TOTAL_BITS_USED;
            let beginning = get_random_key();

            for i in 0..big_limit {
                let a = beginning + i.into();
                let b = beginning + (i + 1).into();

                if b > beginning {
                    break;
                }

                let a = Keyyy::from_hilbert_index(a);
                let b = Keyyy::from_hilbert_index(b);

                let thing_1 = Keyyy::z_index(a);
                let thing_2 = Keyyy::from_z_index(thing_1);
                assert_eq!(a, thing_2);

                let thing_1 = Keyyy::hilbert_index(a);
                let thing_2 = Keyyy::from_hilbert_index(thing_1);
                assert_eq!(a, thing_2);

                assert!(are_adjacent(a, b));
            }
        };
    }

    #[test]
    fn u8_17d() {
        check_vals!(u8, 17);
    }

    #[test]
    fn u8_18d() {
        check_vals!(u8, 18);
    }

    #[test]
    fn u8_19d() {
        check_vals!(u8, 19);
    }

    #[test]
    fn u8_20d() {
        check_vals!(u8, 20);
    }

    #[test]
    fn u16_9d() {
        check_vals!(u16, 9);
    }

    #[test]
    fn u16_10d() {
        check_vals!(u16, 10);
    }

    #[test]
    fn u16_11d() {
        check_vals!(u16, 11);
    }

    #[test]
    fn u32_5d() {
        check_vals!(u32, 5);
    }

    #[test]
    fn u32_6d() {
        check_vals!(u32, 6);
    }

    #[test]
    fn u32_7d() {
        check_vals!(u32, 7);
    }

    #[test]
    fn u64_3d() {
        check_vals!(u64, 3);
    }

    #[test]
    fn u64_5d() {
        check_vals!(u64, 5);
    }
}
