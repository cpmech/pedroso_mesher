//! Exact signs of determinants whose inputs are binary64 numbers.
//!
//! A binary64 value is an integer multiplied by a power of two. Scaling every
//! coordinate by the same power of two therefore turns the determinants into
//! integer expressions without changing their signs. The small signed integer
//! below uses base 2^32 and provides only the operations needed here.

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Integer {
    negative: bool,
    digits: Vec<u32>,
}

impl Integer {
    pub(crate) fn zero() -> Self {
        Self {
            negative: false,
            digits: Vec::new(),
        }
    }

    fn from_u64(value: u64) -> Self {
        if value == 0 {
            return Self::zero();
        }
        Self {
            negative: false,
            digits: vec![value as u32, (value >> 32) as u32],
        }
        .normalized()
    }

    fn from_digits(digits: Vec<u32>, negative: bool) -> Self {
        Self { negative, digits }.normalized()
    }

    fn normalized(mut self) -> Self {
        while self.digits.last() == Some(&0) {
            self.digits.pop();
        }
        if self.digits.is_empty() {
            self.negative = false;
        }
        self
    }

    pub(crate) fn sign(&self) -> i8 {
        if self.digits.is_empty() {
            0
        } else if self.negative {
            -1
        } else {
            1
        }
    }

    fn magnitude_cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.digits.len().cmp(&other.digits.len()) {
            std::cmp::Ordering::Equal => self.digits.iter().rev().cmp(other.digits.iter().rev()),
            ordering => ordering,
        }
    }

    fn negated(&self) -> Self {
        if self.digits.is_empty() {
            return Self::zero();
        }
        Self {
            negative: !self.negative,
            digits: self.digits.clone(),
        }
    }

    pub(crate) fn add(&self, other: &Self) -> Self {
        if self.digits.is_empty() {
            return other.clone();
        }
        if other.digits.is_empty() {
            return self.clone();
        }
        if self.negative == other.negative {
            return Self::from_digits(add_magnitudes(&self.digits, &other.digits), self.negative);
        }
        match self.magnitude_cmp(other) {
            std::cmp::Ordering::Greater => Self::from_digits(
                subtract_magnitudes(&self.digits, &other.digits),
                self.negative,
            ),
            std::cmp::Ordering::Less => Self::from_digits(
                subtract_magnitudes(&other.digits, &self.digits),
                other.negative,
            ),
            std::cmp::Ordering::Equal => Self::zero(),
        }
    }

    pub(crate) fn subtract(&self, other: &Self) -> Self {
        self.add(&other.negated())
    }

    pub(crate) fn multiply(&self, other: &Self) -> Self {
        if self.digits.is_empty() || other.digits.is_empty() {
            return Self::zero();
        }
        let mut digits = vec![0u32; self.digits.len() + other.digits.len()];
        for (i, &left) in self.digits.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &right) in other.digits.iter().enumerate() {
                let Some(slot) = digits.get_mut(i + j) else {
                    return Self::zero();
                };
                let value = u64::from(left) * u64::from(right) + u64::from(*slot) + carry;
                *slot = value as u32;
                carry = value >> 32;
            }
            let mut position = i + other.digits.len();
            while carry != 0 {
                let Some(slot) = digits.get_mut(position) else {
                    return Self::zero();
                };
                let value = u64::from(*slot) + carry;
                *slot = value as u32;
                carry = value >> 32;
                position += 1;
            }
        }
        Self::from_digits(digits, self.negative != other.negative)
    }

    fn times_power_of_two(&self, bits: usize) -> Self {
        if self.digits.is_empty() || bits == 0 {
            return self.clone();
        }
        let whole_digits = bits / 32;
        let remaining = bits % 32;
        let mut digits = vec![0u32; whole_digits + self.digits.len() + 1];
        for (i, &digit) in self.digits.iter().enumerate() {
            let shifted = u64::from(digit) << remaining;
            let Some(low) = digits.get_mut(i + whole_digits) else {
                return Self::zero();
            };
            *low |= shifted as u32;
            let Some(high) = digits.get_mut(i + whole_digits + 1) else {
                return Self::zero();
            };
            *high |= (shifted >> 32) as u32;
        }
        Self::from_digits(digits, self.negative)
    }
}

fn add_magnitudes(left: &[u32], right: &[u32]) -> Vec<u32> {
    let count = left.len().max(right.len());
    let mut result = Vec::with_capacity(count + 1);
    let mut carry = 0u64;
    for i in 0..count {
        let a = left.get(i).copied().unwrap_or(0);
        let b = right.get(i).copied().unwrap_or(0);
        let sum = u64::from(a) + u64::from(b) + carry;
        result.push(sum as u32);
        carry = sum >> 32;
    }
    if carry != 0 {
        result.push(carry as u32);
    }
    result
}

/// Subtract two magnitudes, requiring `left >= right`.
fn subtract_magnitudes(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(left.len());
    let mut borrow = 0i64;
    for (i, &a) in left.iter().enumerate() {
        let b = right.get(i).copied().unwrap_or(0);
        let difference = i64::from(a) - i64::from(b) - borrow;
        if difference < 0 {
            result.push((difference + (1i64 << 32)) as u32);
            borrow = 1;
        } else {
            result.push(difference as u32);
            borrow = 0;
        }
    }
    result
}

/// Return the sign, 53-bit significand and binary exponent of a finite value.
fn decompose(value: f64) -> Option<(bool, u64, i32)> {
    if !value.is_finite() || value == 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let stored_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if stored_exponent == 0 {
        Some((negative, fraction, -1074))
    } else {
        Some((
            negative,
            fraction | (1u64 << 52),
            stored_exponent - 1023 - 52,
        ))
    }
}

fn scaled_integers(values: &[f64]) -> Option<Vec<Integer>> {
    let mut common_exponent = i32::MAX;
    for &value in values {
        if !value.is_finite() {
            return None;
        }
        if let Some((_, _, exponent)) = decompose(value) {
            common_exponent = common_exponent.min(exponent);
        }
    }
    if common_exponent == i32::MAX {
        return Some(vec![Integer::zero(); values.len()]);
    }
    let mut integers = Vec::with_capacity(values.len());
    for &value in values {
        let integer = match decompose(value) {
            None if value == 0.0 => Integer::zero(),
            None => return None,
            Some((negative, significand, exponent)) => {
                let shift = usize::try_from(exponent - common_exponent).ok()?;
                let magnitude = Integer::from_u64(significand).times_power_of_two(shift);
                if negative {
                    magnitude.negated()
                } else {
                    magnitude
                }
            }
        };
        integers.push(integer);
    }
    Some(integers)
}

pub(crate) fn orientation(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i8 {
    let coordinates = [a[0], a[1], b[0], b[1], c[0], c[1]];
    let Some(v) = scaled_integers(&coordinates) else {
        return 0;
    };
    if v.len() != 6 {
        return 0;
    }
    let ax = v[0].subtract(&v[4]);
    let ay = v[1].subtract(&v[5]);
    let bx = v[2].subtract(&v[4]);
    let by = v[3].subtract(&v[5]);
    ax.multiply(&by).subtract(&ay.multiply(&bx)).sign()
}

pub(crate) fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> i8 {
    let coordinates = [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]];
    let Some(v) = scaled_integers(&coordinates) else {
        return 0;
    };
    if v.len() != 8 {
        return 0;
    }
    let adx = v[0].subtract(&v[6]);
    let ady = v[1].subtract(&v[7]);
    let bdx = v[2].subtract(&v[6]);
    let bdy = v[3].subtract(&v[7]);
    let cdx = v[4].subtract(&v[6]);
    let cdy = v[5].subtract(&v[7]);

    let ab = adx.multiply(&bdy).subtract(&bdx.multiply(&ady));
    let bc = bdx.multiply(&cdy).subtract(&cdx.multiply(&bdy));
    let ca = cdx.multiply(&ady).subtract(&adx.multiply(&cdy));
    let aa = adx.multiply(&adx).add(&ady.multiply(&ady));
    let bb = bdx.multiply(&bdx).add(&bdy.multiply(&bdy));
    let cc = cdx.multiply(&cdx).add(&cdy.multiply(&cdy));
    aa.multiply(&bc)
        .add(&bb.multiply(&ca))
        .add(&cc.multiply(&ab))
        .sign()
}

pub(crate) fn encroachment_sign(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> i8 {
    let coordinates = [a[0], a[1], b[0], b[1], p[0], p[1]];
    let Some(v) = scaled_integers(&coordinates) else {
        return 0;
    };
    if v.len() != 6 {
        return 0;
    }
    let ax = v[0].subtract(&v[4]);
    let ay = v[1].subtract(&v[5]);
    let bx = v[2].subtract(&v[4]);
    let by = v[3].subtract(&v[5]);
    ax.multiply(&bx).add(&ay.multiply(&by)).sign()
}

pub(crate) fn ring_area(ring: &[[f64; 2]]) -> i8 {
    if ring.len() < 3 {
        return 0;
    }
    let mut coordinates = Vec::with_capacity(ring.len() * 2);
    for point in ring {
        coordinates.extend_from_slice(point);
    }
    let Some(v) = scaled_integers(&coordinates) else {
        return 0;
    };
    let mut total = Integer::zero();
    for i in 0..ring.len() {
        let j = (i + 1) % ring.len();
        let (Some(ax), Some(ay), Some(bx), Some(by)) = (
            v.get(2 * i),
            v.get(2 * i + 1),
            v.get(2 * j),
            v.get(2 * j + 1),
        ) else {
            return 0;
        };
        let term = ax.multiply(by).subtract(&bx.multiply(ay));
        total = total.add(&term);
    }
    total.sign()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_integer_arithmetic_crosses_digit_boundaries() {
        let max = Integer::from_u64(u64::MAX);
        let two = Integer::from_u64(2);
        assert_eq!(max.add(&two).subtract(&two), max);
        assert_eq!(two.subtract(&max).sign(), -1);
        assert_eq!(max.multiply(&two).subtract(&max), max);
        assert_eq!(max.subtract(&max).sign(), 0);
    }

    #[test]
    fn exact_encroachment_distinguishes_inside_on_and_outside() {
        assert_eq!(encroachment_sign([0.0, 0.0], [2.0, 0.0], [1.0, 0.5]), -1);
        assert_eq!(encroachment_sign([0.0, 0.0], [2.0, 0.0], [1.0, 1.0]), 0);
        assert_eq!(encroachment_sign([0.0, 0.0], [2.0, 0.0], [3.0, 0.0]), 1);
    }

    #[test]
    fn binary64_decomposition_handles_subnormals() {
        let value = f64::from_bits(1);
        let integers = scaled_integers(&[value, 0.0]);
        assert!(integers.is_some());
        assert_eq!(orientation([0.0, 0.0], [value, 0.0], [0.0, value]), 1);
    }
}
