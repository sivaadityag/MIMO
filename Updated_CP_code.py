approx = ComplexField(32)  # precision


class GRSBase:
    """GRS subcode for CP code.
    Note: This works for non-prime fields, unlike the CP code subclass.
    Parameters:
        length - length of the code; this is equal to q - 1, where q is the
            size of the base field
        k - degree of the CP message polynomial (paper's k).
            The associated GRS code GRS(F(k-1,q)) has dimension k.
        chi_ind (optional, 1 by default) - index of the character function,
            only used in the CP code subclass
    """
    def __init__(self, length, k, chi_ind=1):
        # parameters
        self.length = length
        self.k = k
        # finite field
        self.field_size = length + 1
        F_q.<w> = GF(self.field_size, modulus="primitive")
        self.field = F_q
        self.field_char = self.field.characteristic()
        self.field_units = [w^i for i in range(length)]
        self.chi_ind = chi_ind % self.field_char
        # polynomial ring
        F_q_x.<x> = self.field[]
        self.poly_ring = F_q_x
        # GRS code: GRS(F(k-1, q)) has dimension k
        self.grs = codes.GeneralizedReedSolomonCode(
            self.field_units,
            self.k,
            self.field_units
        )
        self.grs_decoder = codes.decoders.GRSBerlekampWelchDecoder
        self.grs_list_decoder = codes.decoders.GRSGuruswamiSudanDecoder
        # pre-compute dimension and minimum distance for __str__ and __eq__
        self._dim = self.dimension()
        self._dmin = self.minimum_distance()

    def __str__(self):
        return (f"({self.length}, {self._dim}, {self._dmin}) CP code with "
                f"character chi_{self.chi_ind} associated with {self.grs}")

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, GRSBase):
            return (self._dim == other._dim and self._dmin == other._dmin)
        return NotImplemented

    def polynomial_ring(self):
        """Return the polynomial ring of self."""
        return self.poly_ring

    def __grs_generator_matrix(self):
        """Return a generator matrix of the GRS subcode of self.

        The GRS code encodes g in F(k-1, q) as (alpha_i * g(alpha_i)).
        We need g = f/X where f in F_p(k, q), so g_i = f_{i+1}.
        Forced zeros in f: indices 0, p, 2p, ...
        Forced zeros in g: indices where (i+1) % p == 0, i.e., i = p-1, 2p-1, ...
        """
        mat = copy(self.grs.generator_matrix())
        p = self.field_char
        for i in range(self.k):
            if (i + 1) % p == 0:
                mat[i,:] = 0
        return mat

    def grs_subcode(self):
        """Return the GRS subcode of self.grs."""
        code = LinearCode(self.__grs_generator_matrix())
        print(code)
        return code
        # return LinearCode(self.__grs_generator_matrix())

    def dimension(self):
        return self.grs_subcode().dimension()

    def minimum_distance(self):
        return self.grs_subcode().minimum_distance()

    def covering_radius(self):
        return self.grs_subcode().covering_radius()


class CharacterPolynomialCode(GRSBase):
    def __init__(self, length, k, chi_ind=1):
        if not is_prime(length + 1):
            raise ValueError("This implementation only works for prime fields")
        super().__init__(length, k, chi_ind)
        # pre-compute character function and its inverse
        self.__chi_mem = dict()
        self.__phi_mem = dict()
        for a in self.field:
            self.__chi_mem[a] = self.__chi(a)
            self.__phi_mem[self.__chi_mem[a]] = a

    def __contains__(self, c):
        """
        If c is a polynomial, return True iff c is in the message space of self.
        If c is a vector, return True iff c is a valid codeword of self.
        """
        if isinstance(
            c,
            sage.rings.polynomial.polynomial_zmod_flint.Polynomial_zmod_flint
        ):
            for i, coeff in enumerate(c.list()):
                if i % self.field_char == 0 and coeff != 0:
                    return False
            return c in self.poly_ring and c.degree() < self.k + 1
        if isinstance(
            c,
            sage.modules.free_module_element.FreeModuleElement_generic_dense
        ):
            try:
                f = self.decode(c)
                return self.encode(f) == c
            except sage.coding.decoder.DecodingError:
                return False
        return False

    def dimension(self):
        """Paper's equation (3): dimension = k - floor(k/p)"""
        return self.k - self.k // self.field_char

    def minimum_distance(self):
        """GRS(F(k-1,q)) has d = n - k + 1"""
        return self.length - self.k + 1

    def covering_radius(self):
        return self.minimum_distance() - 1

    def decoding_radius(self):
        return self.grs_decoder(self.grs).decoding_radius()

    def list_decoding_radius(self):
        return self.grs_list_decoder.guruswami_sudan_decoding_radius(self.grs)[0]

    def convert_polynomial(self, f):
        """Convert f to a polynomial in the message space of self by setting
        every p-th coefficient equal to 0, where p = self.field_char.
        """
        if not (f in self.poly_ring and f.degree() < self.k + 1):
            raise ValueError(f"{f} is not an element of {self.poly_ring} "
                             f"of degree less than {self.k + 1}")
        coeffs = f.list()
        for i in range(len(coeffs)):
            if i % self.field_char == 0:
                coeffs[i] = 0
        return self.poly_ring(coeffs)

    def __chi(self, a):
        """Compute the additive character of self.field corresponding to a."""
        if not a in self.field:
            raise ValueError(f"{a} is not an element of {self.field}")
        ja = self.chi_ind * a
        return exp(2*pi*I*lift(ja.trace())/self.field_char)

    def __encode_raw(self, f):
        """Encode f directly, i.e. without using the built-in GRS encoder."""
        cp_rs_cw = vector(
            self.field,
            [a * f(a) for a in self.field_units]
        )
        cp_vec = [self.__chi_mem[c] for c in cp_rs_cw]
        return vector(approx, cp_vec)

    def encode(self, f, debug=False):
        if not f in self:
            raise ValueError(f"{f} is not an element of the message space "
                             f"of {self}")
        g = f // self.poly_ring.gen()
        cp_rs_cw = self.grs.encoder("EvaluationPolynomial").encode(g)
        if debug:
            print(cp_rs_cw)
        cp_vec = [self.__chi_mem[c] for c in cp_rs_cw]
        cp_cw = vector(approx, cp_vec)
        if debug:
            print("Encoder success?", cp_cw == self.__encode_raw(g))
        return cp_cw

    def __nearest_img(self, z):
        """Return the nearest point (in Euclidean distance) to z in chi(F_q)."""
        p = self.field_char
        z = approx(z)
        if z.arg() >= 0:
            ex = floor(self.length*approx(z.arg())/(2*pi) + 1/2)
        else:
            ex = (p+1)//2 + floor(self.length*approx(pi+z.arg())/(2*pi) + 1/2)
        return exp(2*pi*I/p)^ex

    def __phi(self, z):
        """Return the pre-image of the nearest point to z in chi(F_q)."""
        z_to_unit_circle = self.__nearest_img(z)
        return self.__phi_mem.get(z_to_unit_circle)

    def __pre_process(self, m, debug=False):
        """Return the result of applying phi to each coordinate of m."""
        assert(len(m) == self.length)
        y = [0] * self.length
        for i, mi in enumerate(m):
            y[i] = self.__phi(mi)
        ynew = vector(self.field, y)
        if debug:
            print(f"After phi:\n\t{ynew}")
        return ynew

    def decode(self, m, debug=False):
        if debug:
            print(f"Running CP decoder using {self.grs_decoder} on received "
                  f"message {m}")
        ynew = self.__pre_process(m, debug)
        return (self.poly_ring.gen() *
                self.grs_decoder(self.grs).decode_to_message(ynew))

    def list_decode(self, m, debug=False):
        if debug:
            print(f"Running CP list decoder using {self.grs_list_decoder} on "
                  f"received message {m}")
        ynew = self.__pre_process(m, debug)
        gs_output = self.grs_list_decoder(
            self.grs,
            tau=self.list_decoding_radius()
        ).decode_to_message(ynew)
        grs_list = [self.poly_ring.gen() * f for f in gs_output]
        return [f for f in grs_list if f in self]