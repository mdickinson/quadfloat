Decimal and binary conversions
==============================

For normal Python floats, if we assume that they follow the 53-bit precision
IEEE 754 binary64 format (as they do on the overwhelming majority of
platforms), there are two magic numbers associated with binary to decimal
conversion: when converting binary to decimal, it's sufficient to output 17
digits of precision in order to distinguish distinct floats.  When converting
decimal to binary, distinct 15-digit decimal strings lead to distinct floats.

Where do these numbers (17 and 15) come from?

Notation.  Write R+ for the positive reals.  For any positive integer p, write
Decimal(p) for the subset of R+ consisting of all numbers expressible in the
form n * 10**e for some integers n and e with 0 < n <= 10**p.  Similarly, for
any positive integer q, write Binary(q) for the subset of R+ consisting of all
numbers expressible in the form m * 2**f for some integers m and f satisfying 0
< m <= 2**q.  Let's call these subsets of R+, collectively, 'formats'.

Definition.  For any format F (binary or decimal), there's a map rnd_F : R+ ->
F taking each positive real to the nearest element of F; for concreteness,
assume that ties map to 'even' elements of F in the usual manner (though in
what follows, we don't actually care which way the ties round).

Definition.  For any format F (binary or decimal) and any positive real number
x, we define the quantities ulp_lower_F(x) and ulp_upper_F(x) as follows: if x
is not in F, then both ulp_lower_F(x) and ulp_upper_F(x) are equal to the
difference between the two elements of F neighbouring x.  If x is in F, then
ulp_lower_F(x) is the difference between x and the next smaller element of F,
while ulp_upper_F(x) is the difference between x and the next larger element
of F.  Thus in all cases, ulp_lower_F(x) <= ulp_upper_F(x), and they fail to be
equal only at power of 2 or power of 10 boundaries (as appropriate).

The following statements are then straightforward to justify.

Proposition.  For any format F (binary or decimal) and any real number x,
|x - rnd_F(x)| <= 1/2 ulp_upper_F(x).

Proposition.  Let F be a binary or decimal format and suppose x is in F.
Then for any y in R+ satisfying |x - y| < 1/2 ulp_lower_F(x), rnd(y) = x.

Proposition.  If D = Decimal(p) and B = Binary(q) satisfy 10**p < 2**(q-1)
then for any positive real x, ulp_upper_B(x) < ulp_lower_D(x).

Proof. Choose integers e and f such that 2**(e-1) <= x < 2**e and 10**(f-1) < x
<= 10**f.  Then 2**(e-1) <= x <= 10**f, so

  ulp_upper_B(x) = 2**(e-q) = 2**(e-1) / 2**(q-1)
                 <= 10**f / 2**(q-1)
                 < 10**f / 10**p  (since 10**p < 2**(q-1) by assumption)
                 = 10**(f - p) = ulp_lower_D(x).

Theorem.  If D = Decimal(p) and B = Binary(q) satisfy 10**p < 2**(q-1)
then for any element x of D, rnd_D(rnd_B(x)) == x.

Proof.  Combine the propositions above.

Corollary.  With the conditions of the theorem, rnd_B : D -> B is injective.

The same argument with D and B reversed shows:

Theorem.  If D = Decimal(p) and B = Binary(q) satisfy 2**q < 10**(p-1)
then for any element x of B, rnd_B(rnd_D(x)) = x.


So where do 17 and 15 come from?  17 is the smallest p satisfying 2**53 <
10**(p-1), while 15 is the largest p satisfying 10**p < 2**52.  That is,
17 = ceil(53 * log10(2)) + 1, and 15 = floor(52 * log10(2)).

The corresponding values for single precision are 6 and 9.  That is, 9 digits
are necessary to accurately represent a single-precision float, while 6 decimal
digits are the most that can be faithfully represented in a single-precision
float.  To see this, consider:

    0x1.99999ap-4 = 0.100000001490116119384765625
    0x1.99999cp-4 = 0.10000000894069671630859375
    0x1.99999ep-4 = 0.100000016391277313232421875
    0x1.999a0ep-4 = 0.10000002384185791015625

With only 8 digits of precision, both of the last two values would round to the same
Decimal string.  We need 9 digits to disambiguate.  For the 6 digit claim, consider:

    9.007202e15 rounds to 0x1.000006p53
    9.007203e15 rounds to 0x1.000006p53.

So 7 digits can't be faithfully represented.
