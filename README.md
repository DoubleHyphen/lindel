# Lindel (lineariser-delineariser)

## Introduction
The `lindel` crate offers functions for transforming arrays of primitive unsigned integers to Morton or Hilbert keys and back, via the eponymous encoding processes. This helps linearise data points while preserving some measure of locality.

This crate is an extension of the `morton-encoding` crate.

## Getting started
If it is not necessary to use `lindel` with `nalgebra`, it is sufficient to insert the line

```toml
lindel = "1.0"
```

under the `[dependencies]` section. Otherwise, the following section must be inserted to the project's `Cargo.toml` file:

```toml
[dependencies.lindel]
version = "1.0"
features = ["nalgebra"]
```

## Usage
### Primitive integers
```rust
use lindel::*;
let input = 99251;
let output_1: [u8; 5] = hilbert_decode(input);
let output_2: [u32; 2] = morton_decode(input);
let input = [543u32, 23765];
let output_1 = input.hilbert_index();
let output_2 = input.z_index();
```
Please note the necessity of specifying the output data-types for the decoding operations.

### `Point`s:
```rust
use nalgebra::Point;
use nalgebra::U4;
use lindel::nalgebra_points::Lineariseable;
type FourDees = Point<u32, U4>;
let input = 26327612u128;
let pnt = FourDees::from_z_index(input);
let result = pnt.hilbert_index();
```

### New large uints:

```rust
lindel::create_lineariseable_data_type!(u128, 33, NewKey);
let input = [870u128; 33];
let hind = NewKey::hilbert_index(input);
let zind = NewKey::z_index(input);
let reinstated_input = hind.from_hilbert_index();
assert_eq!(input, reinstated_input);
let reinstated_input = NewKey::from_z_index(zind);
assert_eq!(input, reinstated_input);
```

## Advantages and Disadvantages
Long story short: Choose Morton encoding (“z-indexing”) if speed is more important than locality. Otherwise, feel free to use Hilbert encoding everywhere.
