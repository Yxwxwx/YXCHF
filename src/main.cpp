#include "Mole.h"



int main(int argc, char* argv[]) {
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    Mole mole;
    auto atoms = mole.read_geometry(filename);
    auto shells = mole.make_sto3g_basis(atoms);

    auto start = std::chrono::high_resolution_clock::now();
    mole.m_S = mole.calc_ovlp(shells);
    mole.m_H = mole.calc_core_h(shells, atoms);
    mole.m_I = mole.calc_eri(shells);
    //std::cout << I.sum() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    
    /*
    std::cout << "S:\n" << S << "\n";
    std::cout << "T:\n" << T << "\n";
    std::cout << "V:\n" << V << "\n";
    */

    // 计算并输出耗时
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time to calculate S: " << diff.count() << " s\n";
    return 0;
}

double Mole::expansion_coefficients(int i, int j, int t, double Qx, double a, double b) {
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
double Mole::overlap_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B) {
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
Matrix Mole::calc_ovlp(std::vector<Shell>& shells) {

    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Matrix S = Matrix::Zero(nao, nao);
    //m_S.setZero(nao, nao);    

    for (auto ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (auto jpr = ipr; jpr < nshls; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
            }
            
            Matrix buf;
            buf.setZero(di, dj);
            
            auto fx = 0;
            for (auto lmn1 : shells[ipr].shl) {
                auto normA = norm(lmn1, shells[ipr].exp);
                auto coeffA = shells[ipr].coeff;
                normalization(coeffA, lmn1, normA, shells[ipr].exp);
                auto fy = 0;
                for (auto lmn2 : shells[jpr].shl) {
                    auto normB = norm(lmn2, shells[jpr].exp);
                    auto coeffB = shells[jpr].coeff;
                    normalization(coeffB, lmn2, normB, shells[jpr].exp);
                    auto ia = 0;
                    for (auto ca : coeffA ) {
                        auto ib = 0;
                        for (auto cb : coeffB) {

                            buf(fx, fy) += normA[ia] * normB[ib] * ca * cb * \
                                        overlap_elem(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                        shells[jpr].exp[ib], lmn2, shells[jpr].coord);
                        ib++;
                        }
                    ia++;
                    }
                fy++;
                }
            fx++;    
            }
            for (int i = 0; i < di; ++i) {
                for (int j = 0; j < dj; ++j) {
                    S(x + i, y + j) = buf(i, j);
                    S(y + j, x + i) = buf(i, j);
                }
            }
            //std::cout << "buf" << buf << std::endl<< std::endl;
            //std::cout <<"S" << S << std::endl << std::endl;
        }  
    }
    
    return S;
}
double Mole::kinetic_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B) {
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
Matrix Mole::calc_kin(std::vector<Shell>& shells) {

    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }
    Matrix T = Matrix::Zero(nao, nao);
    //m_T.setZero(nao, nao);    

    for (auto ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (auto jpr = ipr; jpr < nshls; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
            }
            
            Matrix buf;
            buf.setZero(di, dj);
            
            auto fx = 0;
            for (auto lmn1 : shells[ipr].shl) {
                auto normA = norm(lmn1, shells[ipr].exp);
                auto coeffA = shells[ipr].coeff;
                normalization(coeffA, lmn1, normA, shells[ipr].exp);
                auto fy = 0;
                for (auto lmn2 : shells[jpr].shl) {
                    auto normB = norm(lmn2, shells[jpr].exp);
                    auto coeffB = shells[jpr].coeff;
                    normalization(coeffB, lmn2, normB, shells[jpr].exp);
                    auto ia = 0;
                    for (auto ca : coeffA) {
                        auto ib = 0;
                        for (auto cb : coeffB) {

                            buf(fx, fy) += normA[ia] * normB[ib] * ca * cb * \
                                        kinetic_elem(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                        shells[jpr].exp[ib], lmn2, shells[jpr].coord);
                        ib++;
                        }
                    ia++;
                    }
                fy++;
                }
            fx++;    
            }
            for (int i = 0; i < di; ++i) {
                for (int j = 0; j < dj; ++j) {
                    T(x + i, y + j) = buf(i, j);
                    T(y + j, x + i) = buf(i, j);
                }
            }
        }  
    }
    
    return T;
}
double Mole::coulomb_auxiliary_hermite_integrals(int t, int u, int v, double n, double p, double PCx, double PCy, double PCz, double RPC) {
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
double Mole::nuclear_elem(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B, std::vector<double>& C) {
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

Matrix Mole::calc_nuc(std::vector<Shell>& shells, std::vector<Atom>& atoms) {

    std::vector<std::pair<int, std::vector<double>>> q;
    for (const auto& atom : atoms) {
         q.push_back({static_cast<int>(atom.atomic_number),
                   {{atom.x, atom.y, atom.z}}});
    }

    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Matrix V = Matrix::Zero(nao, nao);
    //m_V.setZero(nao, nao);    

    for (auto ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (auto jpr = ipr; jpr < nshls; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (auto j = 0; j < jpr; j++) {
                y += shells[j].ngto;
            }
            
            Matrix buf;
            buf.setZero(di, dj);
            
            auto fx = 0;
            for (auto lmn1 : shells[ipr].shl) {
                auto normA = norm(lmn1, shells[ipr].exp);
                auto coeffA = shells[ipr].coeff;
                normalization(coeffA, lmn1, normA, shells[ipr].exp);
                auto fy = 0;
                for (auto lmn2 : shells[jpr].shl) {
                    auto normB = norm(lmn2, shells[jpr].exp);
                    auto coeffB = shells[jpr].coeff;
                    normalization(coeffB, lmn2, normA, shells[ipr].exp);
                    auto ia = 0;
                    for (auto ca : coeffA) {
                        auto ib = 0;
                        for (auto cb : coeffB) { 
                            for (auto nuc_cent : q) {   

                                buf(fx, fy) += -nuc_cent.first * normA[ia] * normB[ib] * ca * cb * \
                                        nuclear_elem(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                        shells[jpr].exp[ib], lmn2, shells[jpr].coord, nuc_cent.second);
                            
                            }  
                        ib++;
                        }
                    ia++;
                    }
                fy++;
                }
            fx++;    
            }
            for (int i = 0; i < di; ++i) {
                for (int j = 0; j < dj; ++j) {
                    V(x + i, y + j) = buf(i, j);
                    V(y + j, x + i) = buf(i, j);
                }
            }
        }  
    }
    
    return V;
}
double Mole::electron_repulsion(double a, std::vector<int>& lmn1, std::vector<double>& A, double b, std::vector<int>& lmn2, std::vector<double>& B, double c, std::vector<int>& lmn3,std::vector<double>& C, double d, std::vector<int>& lmn4,std::vector<double>& D) {
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

Tensor Mole::calc_eri(std::vector<Shell>& shells) {
    auto nshls = shells.size();
    auto nao = 0;
    for (const auto& shell : shells) {
        nao += shell.ngto;
    }

    Tensor I(nao, nao, nao ,nao);
    I.setZero();
    for (int ipr = 0; ipr < nshls; ipr++) {
        auto di = shells[ipr].ngto;
        auto x = 0;
        for (int i = 0; i < ipr; i++) {
            x += shells[i].ngto;
        }
        for (int jpr = 0; jpr <= ipr; jpr++) {
            auto dj = shells[jpr].ngto;
            auto y = 0;
            for (int i = 0; i < jpr; i++) {
            y += shells[i].ngto;
            }
            for (int kpr = 0; kpr <= ipr; kpr++) {
                auto dk = shells[kpr].ngto;
                auto z = 0;
                for (int i = 0; i < kpr; i++) {
                    z += shells[i].ngto;
                }
                const auto lpr_max = (ipr == kpr) ? jpr : kpr;
                for (int lpr = 0; lpr <=  lpr_max; lpr++) {
                    auto dl = shells[lpr].ngto;
                    auto w = 0;
                    for (int i = 0; i < lpr; i++) {
                        w += shells[i].ngto;
                    }
                    Tensor buf(di, dj, dk, dl);
                    buf.setZero();
                    //std::cout << "(" << ipr << jpr <<"|" << kpr << lpr << ")" << std::endl;

                    auto fx = 0;
                    for (auto lmn1 : shells[ipr].shl) {
                        auto normA = norm(lmn1, shells[ipr].exp);
                        auto coeffA = shells[ipr].coeff;
                        normalization(coeffA, lmn1, normA, shells[ipr].exp);
                        auto fy = 0;
                        for (auto lmn2 : shells[jpr].shl) {
                            auto normB = norm(lmn2, shells[jpr].exp);
                            auto coeffB = shells[jpr].coeff;
                            normalization(coeffB, lmn2, normB, shells[jpr].exp);
                            auto fz = 0;
                            for (auto lmn3 : shells[kpr].shl) {
                                auto normC = norm(lmn3, shells[kpr].exp);
                                auto coeffC = shells[kpr].coeff;
                                normalization(coeffC, lmn3, normC, shells[kpr].exp);
                                auto fw = 0;
                                for (auto lmn4 : shells[lpr].shl) {
                                    auto normD = norm(lmn4, shells[lpr].exp);
                                    auto coeffD = shells[lpr].coeff;
                                    normalization(coeffD, lmn4, normD, shells[lpr].exp);
                                    
                                    auto ia = 0;
                                    for (auto ca : coeffA) {
                                        auto ib = 0;
                                        for (auto cb : coeffB) { 
                                            auto ic = 0;
                                            for (auto cc : coeffC) {
                                                auto id = 0;
                                                for (auto cd : coeffD) {
                                                    buf(fx, fy, fz, fw) += normA[ia] * normB[ib] * normC[ic] * normD[id] * \
                                                                            ca * cb * cc * cd * \
                                                                            electron_repulsion(shells[ipr].exp[ia], lmn1, shells[ipr].coord,
                                                                                               shells[jpr].exp[ib], lmn2, shells[jpr].coord,
                                                                                               shells[kpr].exp[ic], lmn3, shells[kpr].coord,
                                                                                               shells[lpr].exp[id], lmn4, shells[lpr].coord);
                                                id++;    
                                                }
                                            ic++;    
                                            }
                                        ib++;
                                        }
                                    ia++;
                                    }
                                fw++;
                                }
                            fz++;
                            }
                        fy++;
                        }
                    fx++;
                    }  
                           
                    for (int i = 0; i < di; ++i) {
                        for (int j = 0; j < dj; ++j) {
                            for (int k = 0; k < dk; ++k) {
                                for (int l = 0; l < dl; l++) {
                                    I(x+i, y+j, z+k, w+l) = buf(i, j, k, l);
                                    I(y+j, x+i, z+k, w+l) = buf(i, j, k, l);
                                    I(x+i, y+j, w+l, z+k) = buf(i, j, k, l);
                                    I(y+j, x+i, w+l, z+k) = buf(i, j, k, l);
                                    I(z+k, w+l, x+i, y+j) = buf(i, j, k, l);
                                    I(w+l, z+k, x+i, y+j) = buf(i, j, k, l);
                                    I(z+k, w+l, y+j, x+i) = buf(i, j, k, l);
                                    I(w+l, z+k, y+j, x+i) = buf(i, j, k, l);
                                }
                            }
                        }  
                    }
                }
            } 
        }
    }
    
    return I;
}