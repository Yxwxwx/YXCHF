#define EIGEN_USE_BLAS
#include "Eigen/Dense"
#include "EigenUnsupported/Eigen/CXX11/Tensor"
// standard C++ headers
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <omp.h>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix; 
typedef Eigen::Tensor<double, 4, Eigen::RowMajor> Tensor;

int fact2(int n) {
    int result = 1;
    while (n > 0) {
        result *= n;
        n -= 2;
    }
    return result;
}
void tensor_print(Tensor& tensor)
{
	//just debug
	int dim0 = tensor.dimension(0);
	int dim1 = tensor.dimension(1);
	int dim2 = tensor.dimension(2);
	int dim3 = tensor.dimension(3);

	for (int i = 0; i < dim0; i++)
	{
		for (int j = 0; j < dim1; j++)
		{
			for (int k = 0; k < dim2; k++)
			{
				for (int l = 0; l < dim3; l++)
				{
					std::cout << "\t" << tensor(i, j, k, l);
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
}

class Mole {
public:

  struct Atom {
    int atomic_number;
    double x, y, z;
  };
  struct Shell {
    int l;  // angular momentum
    std::vector<double> exp;  // exponents of primitive Gaussians
    std::vector<double> coeff;  // coefficients of primitive Gaussians
    std::vector<double> coord;  // origin coordinates
    int ngto;  // number of Gaussian Type Orbitals
    std::vector<std::vector<int>> shl;  // shell

    Shell(int l, std::vector<double> exp, std::vector<double> coeff, std::vector<double> coord)
    : l(l), exp(exp), coeff(coeff), coord(coord), ngto((l+1)*(l+2)/2) {
        switch (l) {
            case 0:
                shl = {{0, 0, 0}};
                break;
            case 1:
                shl = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                break;
            case 2:
                shl = {{1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2}};
                break;
            default:
                throw std::invalid_argument{"do not support angular momentum l"};
        }
    }
};

  int m_charge{ 0 }, m_spin_multiplicity{ 1 };
  int m_natom{ 0 }, m_nelec{ 0 }; // total number of atom/electron

  std::vector<Atom> read_dotxyz(std::istream& is) {
    // line 1 = # of atoms
    size_t natom;
    is >> natom;
    m_natom = natom;
    // read off the rest of line 1 and discard
    std::string rest_of_line;
    std::getline(is, rest_of_line);

    // line 2 = comment (possibly empty)
    std::string comment;
    std::getline(is, comment);

    std::vector<Atom> atoms(natom);
    for (auto i = 0; i < natom; i++) {
      std::string element_label;
      double x, y, z;
      is >> element_label >> x >> y >> z;

      // .xyz files report element labels, hence convert to atomic numbers
      int Z;
      if (element_label == "H")
        Z = 1;
      else if (element_label == "C")
        Z = 6;
      else if (element_label == "N")
        Z = 7;
      else if (element_label == "O")
        Z = 8;
      else if (element_label == "F")
        Z = 9;
      else if (element_label == "S")
        Z = 16;
      else if (element_label == "Cl")
        Z = 17;
      else {
        std::cerr << "read_dotxyz: element label \"" << element_label
                  << "\" is not recognized" << std::endl;
        throw std::invalid_argument(
            "Did not recognize element label in .xyz file");
      }

      atoms[i].atomic_number = Z;
      m_nelec+=Z;
      // .xyz files report Cartesian coordinates in angstroms; convert to bohr
      const auto angstrom_to_bohr = 1 / 0.52917721092;  // 2010 CODATA value
      atoms[i].x = x * angstrom_to_bohr;
      atoms[i].y = y * angstrom_to_bohr;
      atoms[i].z = z * angstrom_to_bohr;
    }

    return atoms;
}
  std::vector<Atom> read_geometry(const std::string& filename) {
    std::cout << "Will read geometry from " << filename << std::endl;
    std::ifstream is(filename);
    assert(is.good());

    // to prepare for MPI parallelization, we will read the entire file into a
    // string that can be broadcast to everyone, then converted to an
    // std::istringstream object that can be used just like std::ifstream
    std::ostringstream oss;
    oss << is.rdbuf();
    // use ss.str() to get the entire contents of the file as an std::string
    // broadcast
    // then make an std::istringstream in each process
    std::istringstream iss(oss.str());

    // check the extension: if .xyz, assume the standard XYZ format, otherwise
    // throw an exception
    if (filename.rfind(".xyz") != std::string::npos)
      return read_dotxyz(iss);
    else
      throw std::invalid_argument("only .xyz files are accepted");
}
  std::vector<Shell> make_sto3g_basis(const std::vector<Atom>& atoms) {

    std::vector<Shell> shells;

    for (auto a = 0; a < atoms.size(); a++)
    {
      switch (atoms[a].atomic_number)
      {
      case 1:
        shells.push_back(Shell(
          0,
          {3.425250910, 0.623913730,0.168855400},  // exponents of primitive Gaussians
          {0.15432897, 0.53532814, 0.44463454}, // coefficients
          {atoms[a].x, atoms[a].y, atoms[a].z} // origin coordinates
        ));
        break;

      case 8:  // Z=8: oxygen
    shells.push_back(Shell(
        0,
        {130.709320000, 23.808861000, 6.443608300},  // exponents of primitive Gaussians
        {0.15432897, 0.53532814, 0.44463454}, // coefficients
        {atoms[a].x, atoms[a].y, atoms[a].z} // origin coordinates
    ));
    shells.push_back(Shell(
        0,
        {5.033151300, 1.169596100, 0.380389000},  // exponents of primitive Gaussians
        {-0.09996723, 0.39951283, 0.70011547}, // coefficients
        {atoms[a].x, atoms[a].y, atoms[a].z} // origin coordinates
    ));
    shells.push_back(Shell(
        1,
        {5.033151300, 1.169596100, 0.380389000},  // exponents of primitive Gaussians
        {0.15591627, 0.60768372, 0.39195739}, // coefficients
        {atoms[a].x, atoms[a].y, atoms[a].z} // origin coordinates
    ));
      break;

      
      default:
        throw std::invalid_argument{"do not know STO-3G basis for this Z"};
      }
    }
    return shells;  // return the shells
} 
  Matrix m_S, m_T, m_V, m_H, m_G; // electric integral
  Tensor m_I; // int2e
  Matrix m_F; //Fock matrix

  std::vector<double> norm(std::vector<int>& lmn,std::vector<double>& exps)
  {
    std::vector<double> norms(exps.size());
    for (int i = 0; i < exps.size(); i++) {
      auto l = lmn[0], m = lmn[1], n = lmn[2];
      norms[i] = std::sqrt(std::pow(2, 2 * (l + m + n) + 1.5) *
                             std::pow(exps[i], l + m + n + 1.5) /
                             fact2(2 * l - 1) / fact2(2 * m - 1) / fact2(2 * n - 1) /
                             std::pow(M_PI, 1.5));
    }
    return norms;
  }
  void normalization (std::vector<double> coeff, std::vector<int>& lmn, std::vector<double> norm, std::vector<double> exp) {
    auto l = lmn[0], m = lmn[1], n = lmn[2];
    auto L = l + m + n;
    double prefactor = std::pow(M_PI, 1.5) *
                       fact2(2 * l - 1) * fact2(2 * m - 1) * fact2(2 * n - 1) /
                       std::pow(2.0, L);
    double N = 0.0;
    for (int i = 0; i < coeff.size(); i++) {
      for (int j = 0; j < coeff.size(); j++) {
        N += norm[i] * norm[j] * coeff[i] * coeff[j] /
             std::pow(exp[i] + exp[j], L + 1.5);
      }
    }

    N *= prefactor;
    N = std::pow(N, -0.5);

    for (int i = 0; i < coeff.size(); i++) {
      coeff[i] *= N;
    }
    
    
  }
  double expansion_coefficients(int i, int j, int t, double Qx, double a, double b) {
    /*
        Recursive definition of Hermite Gaussian coefficients.
        Returns a double.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral, 
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    */
   auto p = a+b;
   auto q = a*b/p;
   if (t < 0 || t > (i + j))
   {
    // out of bounds for t
    return 0.0;
   }
   else if (i == 0 && j == 0 && t == 0)
   {
        return std::exp(-q*Qx*Qx); // K_AB
   }
   else if (j == 0)
   {
        // decrement index i
        return (1/(2*p))*expansion_coefficients(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*expansion_coefficients(i-1,j,t,Qx,a,b)    + \
               (t+1)*expansion_coefficients(i-1,j,t+1,Qx,a,b);
   }
   else
   {
        // decrement index j
        return (1/(2*p))*expansion_coefficients(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*expansion_coefficients(i,j-1,t,Qx,a,b)    + \
               (t+1)*expansion_coefficients(i,j-1,t+1,Qx,a,b);
   }
}
  double overlap_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B) {
    /*
        Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    */
    auto l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2];
    auto l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2];

    double S1 = expansion_coefficients(l1,l2,0,A[0]-B[0],a,b); // X
    double S2 = expansion_coefficients(m1,m2,0,A[1]-B[1],a,b); // Y
    double S3 = expansion_coefficients(n1,n2,0,A[2]-B[2],a,b); // Z
    
    return S1*S2*S3*pow(M_PI/(a+b),1.5);
}

  double kinetic_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B) {
    auto l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2];
    auto l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2];

    std::vector<int> lmn1_2 = {l1, m1, n1};
    std::vector<int> lmn2_2 = {l2 + 2, m2, n2};
    std::vector<int> lmn2_3 = {l2, m2 + 2, n2};
    std::vector<int> lmn2_4 = {l2, m2, n2 + 2};
    std::vector<int> lmn2_5 = {l2 - 2, m2, n2};
    std::vector<int> lmn2_6 = {l2, m2 - 2, n2};
    std::vector<int> lmn2_7 = {l2, m2, n2 - 2};

    double term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap_elem(a, lmn1, A, b, lmn2, B);
    double term1 = -2 * pow(b, 2) * (overlap_elem(a, lmn1_2, A, b, lmn2_2, B) +
                                      overlap_elem(a, lmn1_2, A, b, lmn2_3, B) +
                                      overlap_elem(a, lmn1_2, A, b, lmn2_4, B));
    double term2 = -0.5 * (l2 * (l2 - 1) * overlap_elem(a, lmn1_2, A, b, lmn2_5, B) +
                           m2 * (m2 - 1) * overlap_elem(a, lmn1_2, A, b, lmn2_6, B) +
                           n2 * (n2 - 1) * overlap_elem(a, lmn1_2, A, b, lmn2_7, B));
    return term0 + term1 + term2;
}


double boys(int n, double T) {
    return boost::math::hypergeometric_1F1(n + 0.5, n + 1.5, -T) / (2.0 * n + 1.0);
}

  std::vector<double> gaussian_product_center(double a, const std::vector<double>& A, double b, const std::vector<double>& B) {
    std::vector<double> result;
    for (size_t i = 0; i < A.size(); ++i) {
        result.push_back((a * A[i] + b * B[i]) / (a + b));
    }
    return result;
  }
 double coulomb_auxiliary_hermite_integrals(int t, int u, int v, double n, double p, double PCx, double PCy, double PCz, double RPC) {
    /*
        Returns the Coulomb auxiliary Hermite integrals 
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function 
        PCx,y,z: Cartesian vector distance between Gaussian 
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    */
    auto T = p*RPC*RPC;
    double val = 0.0;
    if (t == 0 && u == 0 && v == 0) {
        val += pow(-2*p, n)*boys(n, T);
    } else if (t == 0 && u == 0) {
        if (v > 1) {
            val += (v-1)*coulomb_auxiliary_hermite_integrals(t, u, v-2, n+1, p, PCx, PCy, PCz, RPC);
        }
        val += PCz*coulomb_auxiliary_hermite_integrals(t, u, v-1, n+1, p, PCx, PCy, PCz, RPC);
    } else if (t == 0) {
        if (u > 1) {
            val += (u-1)*coulomb_auxiliary_hermite_integrals(t, u-2, v, n+1, p, PCx, PCy, PCz, RPC);
        }
        val += PCy*coulomb_auxiliary_hermite_integrals(t, u-1, v, n+1, p, PCx, PCy, PCz, RPC);
    } else {
        if (t > 1) {
            val += (t-1)*coulomb_auxiliary_hermite_integrals(t-2, u, v, n+1, p, PCx, PCy, PCz, RPC);
        }
        val += PCx*coulomb_auxiliary_hermite_integrals(t-1, u, v, n+1, p, PCx, PCy, PCz, RPC);
    }
    return val;
}
double nuclear_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B, std::vector<double>& C) {
    /*
        Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
               for Gaussian 'a'
         lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
         A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
         B:    list containing origin of Gaussian 'b'
         C:    list containing origin of nuclear center 'C'
    */
    auto l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2];
    auto l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2];

    auto p = a + b;
    auto P = gaussian_product_center(a, A, b, B);
    auto RPC = std::sqrt(std::pow(P[0]-C[0], 2) + std::pow(P[1]-C[1], 2) + std::pow(P[2]-C[2], 2));

    double val = 0.0;

    //#pragma omp parallel for reduction(+:val)
    for (int t = 0; t <= l1+l2; t++) {
        for (int u = 0; u <= m1+m2; u++) {
            for (int v = 0; v <= n1+n2; v++) {
                val += expansion_coefficients(l1,l2,t,A[0]-B[0],a,b) *
                       expansion_coefficients(m1,m2,u,A[1]-B[1],a,b) *
                       expansion_coefficients(n1,n2,v,A[2]-B[2],a,b) *
                       coulomb_auxiliary_hermite_integrals(t,u,v,0.0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC);
            }
        }
    }
    val *= 2*M_PI/p;
    return val;
}

  double electron_repulsion(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B, double c, std::vector<int>& lmn3,std::vector<double>& C, double d, std::vector<int>& lmn4,std::vector<double>& D) {
    auto l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2];
    auto l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2];
    auto l3 = lmn3[0], m3 = lmn3[1], n3 = lmn3[2];
    auto l4 = lmn4[0], m4 = lmn4[1], n4 = lmn4[2];

    auto p = a + b;
    auto q = c + d;
    auto alpha = p * q/ (p + q);
    auto P = gaussian_product_center(a, A, b, B);
    auto Q = gaussian_product_center(c, C, d, D);
    auto RPQ = std::sqrt(std::pow(P[0]-Q[0], 2) + std::pow(P[1]-Q[1], 2) + std::pow(P[2]-Q[2], 2));

    double val = 0.0;
    for (int t = 0; t <= l1 + l2; t++) {
        for (int u = 0; u <= m1 + m2; u++) {
            for (int v = 0; v <= n1 + n2; v++) {
                for (int tau = 0; tau <= l3 + l4; tau++) {
                    for (int nu = 0; nu <= m3 + m4; nu++) {
                        for (int phi = 0; phi <= n3 + n4; phi++) {
                            val += expansion_coefficients(l1,l2,t,A[0]-B[0],a,b) *
                                   expansion_coefficients(m1,m2,u,A[1]-B[1],a,b) *
                                   expansion_coefficients(n1,n2,v,A[2]-B[2],a,b) *
                                   expansion_coefficients(l3,l4,tau,C[0]-D[0],c,d) *
                                   expansion_coefficients(m3,m4,nu ,C[1]-D[1],c,d) *
                                   expansion_coefficients(n3,n4,phi,C[2]-D[2],c,d) *
                                   std::pow(-1,tau+nu+phi) *
                                   coulomb_auxiliary_hermite_integrals(t+tau,u+nu,v+phi,0,
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ);
                        }
                    }
                }
            }
        }
    }

    val *= 2 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
    return val;
}
  Matrix calc_ovlp(std::vector<Shell>& shells);
  Matrix calc_kin(std::vector<Shell>& shells);
  Matrix calc_nuc(std::vector<Shell>& shells, std::vector<Atom>& atoms);

  Matrix calc_core_h(std::vector<Shell>& shells, std::vector<Atom>& atoms) {
    m_H = calc_kin(shells) + calc_nuc(shells, atoms);
    return m_H;
  }
  Tensor calc_eri(std::vector<Shell>& shells);
};
