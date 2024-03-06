#define EIGEN_USE_BLAS
#include "Eigen/Dense"
#include "EigenUnsupported/Eigen/CXX11/Tensor"
// standard C++ headers
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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
  double expansion_coefficients(int i, int j, int t, double Qx, double a, double b);
  double overlap_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B);
  Matrix calc_ovlp(std::vector<Shell>& shells);
  double kinetic_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B);
  Matrix calc_kin(std::vector<Shell>& shells);

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
  double coulomb_auxiliary_hermite_integrals(int t, int u, int v, double n, double p, double PCx, double PCy, double PCz, double RPC);
  double nuclear_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B, std::vector<double>& C);
  Matrix calc_nuc(std::vector<Shell>& shells, std::vector<Atom>& atoms);

  Matrix calc_core_h(std::vector<Shell>& shells, std::vector<Atom>& atoms) {
    m_H = calc_kin(shells) + calc_nuc(shells, atoms);
    return m_H;
  }
  double electron_repulsion(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B, double c, std::vector<int>& lmn3,std::vector<double>& C, double d, std::vector<int>& lmn4,std::vector<double>& D);
  Tensor calc_eri(std::vector<Shell>& shells);
};
