//! Implementation of [Compact Hilbert Indices](https://dl.acm.org/doi/10.1109/CISIS.2007.16) by
//! Chris Hamilton, and (in the near future) of Compact Z-indices by the crate maintainer.

/// Right rotation of x by b bits out of n.
const fn rotate_right(x: usize, b: u32, n: u32) -> usize {
    let l = x & ((1 << b) - 1);
    let r = x >> b;
    (l << (n - b)) | r
}

/// Left rotation of x by b bits out of n.
const fn rotate_left(x: usize, b: u32, n: u32) -> usize {
    rotate_right(x, n - b, n)
}

/// Binary reflected Gray code.
const fn gray_code(i: usize) -> usize {
    i ^ (i >> 1)
}

/// e(i), the entry point for the ith sub-hypercube.
const fn entry_point(i: usize) -> usize {
    if i == 0 {
        0
    } else {
        gray_code((i - 1) & !1)
    }
}

/// g(i), the inter sub-hypercube direction.
const fn inter_direction(i: usize) -> u32 {
    // g(i) counts the trailing set bits in i
    (!i).trailing_zeros()
}

/// d(i), the intra sub-hypercube direction.
const fn intra_direction(i: usize) -> u32 {
    if i & 1 != 0 {
        inter_direction(i)
    } else if i > 0 {
        inter_direction(i - 1)
    } else {
        0
    }
}

/// T transformation inverse
const fn t_inverse(dims: u32, e: usize, d: u32, a: usize) -> usize {
    rotate_left(a, d, dims) ^ e
}

/// GrayCodeRankInverse
fn gray_code_rank_inverse(
    dims: u32,
    mu: usize,
    pi: usize,
    r: usize,
    free_bits: u32,
) -> (usize, usize) {
    // The inverse rank of r
    let mut i = 0;
    // gray_code(i)
    let mut g = 0;

    let mut j = free_bits - 1;
    for k in (0..dims).rev() {
        if mu & (1 << k) == 0 {
            g |= pi & (1 << k);
            i |= (g ^ (i >> 1)) & (1 << k);
        } else {
            i |= ((r >> j) & 1) << k;
            g |= (i ^ (i >> 1)) & (1 << k);
            j = j.wrapping_sub(1);
        }
    }

    (i, g)
}

/// ExtractMask.
fn extract_mask(bits: &[u32], i: u32) -> (usize, u32) {
    // The mask
    let mut mu = 0;
    // popcount(mu)
    let mut free_bits = 0;

    let dims = bits.len();
    for j in (0..dims).rev() {
        mu <<= 1;
        if bits[j] > i {
            mu |= 1;
            free_bits += 1;
        }
    }

    (mu, free_bits)
}

/// Compute the corresponding point for a Hilbert index (CompactHilbertIndexInverse).
pub fn from_compact_hilbert_index(index: usize, bits: &[u32], point: &mut [usize]) {
    let dims = bits.len() as u32;
    let max = *bits.iter().max().unwrap();
    let sum: u32 = bits.iter().sum();

    let mut e = 0;
    let mut k = 0;

    // Next direction; we use d instead of d + 1 everywhere
    let mut d = 1;

    for x in point.iter_mut() {
        *x = 0;
    }

    for i in (0..max).rev() {
        let (mut mu, free_bits) = extract_mask(bits, i);
        mu = rotate_right(mu, d, dims);

        let pi = rotate_right(e, d, dims) & !mu;

        let r = (index >> (sum - k - free_bits)) & ((1 << free_bits) - 1);

        k += free_bits;

        let (w, mut l) = gray_code_rank_inverse(dims, mu, pi, r, free_bits);
        l = t_inverse(dims, e, d, l);

        for x in point.iter_mut() {
            *x |= (l & 1) << i;
            l >>= 1;
        }

        e ^= rotate_right(entry_point(w), d, dims);
        d = (d + intra_direction(w) + 1) % dims;
    }
}

/// T transformation
const fn t(dims: u32, e: usize, d: u32, b: usize) -> usize {
    rotate_right(b ^ e, d, dims)
}

/// GrayCodeInverse
const fn gray_code_inverse(mut g: usize) -> usize {
    let mut i = 0;

    while g != 0 {
        i ^= g;
        g >>= 1;
    }

    i
}

/// GrayCodeRank
fn gray_code_rank(dims: u32, mu: usize, i: usize) -> usize {
    let mut r = 0;

    for k in (0..dims).rev() {
        if mu & (1 << k) != 0 {
            r <<= 1;
            r |= (i >> k) & 1;
        }
    }

    r
}

/// CompactHilbertIndex
pub fn compact_hilbert_index(bits: &[u32], point: &[usize]) -> usize {
    let dims = bits.len() as u32;
    let max = *bits.iter().max().unwrap();

    let mut h = 0;
    let mut e = 0;

    // Next direction; we use d instead of d + 1 everywhere
    let mut d = 1;

    for i in (0..max).rev() {
        let (mut mu, free_bits) = extract_mask(bits, i);
        mu = rotate_right(mu, d, dims);

        let mut l = 0;
        for x in point.iter().rev() {
            l <<= 1;
            l |= (x >> i) & 1;
        }
        l = t(dims, e, d, l);

        let w = gray_code_inverse(l);
        let r = gray_code_rank(dims, mu, w);

        e ^= rotate_right(entry_point(w), d, dims);
        d = (d + intra_direction(w) + 1) % dims;
        h = (h << free_bits) | r;
    }

    h
}

/*fn returns_closure (xs: &[u8]) -> impl Fn(u32) -> u32 {
    let sum: u32 = xs.iter().map(|&x| x as u32).sum();
    move |a| a*sum
}

fn main() {
    println!("Hello, world!");
    let fnoo = returns_closure(&[1, 2, 3, 4, 5, 6]);
    dbg!(fnoo(15));
}*/
