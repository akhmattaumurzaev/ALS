#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <complex>
#include <random>



extern "C"
{
    double dznrm2_(const int *, const std::complex<double> *, const int *);
    void zgels_(const char *, const int *, const int *, const int *, std::complex<double> *, const int *, std::complex<double> *, const int *, std::complex<double> *, const int *, int *);
}


void
tensor_make(std::complex<double>** tensor, int length, double& right_side_norm)
{
    int ione = 1;
    int szfull = length;

    *tensor = new std::complex<double>[length];
    for (int i = 0; i < length; i++) {
        std::complex<double> val (-1.0 + 2.0 * double(std::rand()) / RAND_MAX, -1.0 + 2.0 * double(std::rand()) / RAND_MAX);
        (*tensor)[i] = val;
    }

    right_side_norm = dznrm2_(&szfull, (*tensor), &ione);
}

void
tensor_make_not_random(std::complex<double>** tensor, int *razm, int length, int N, int R, double& right_side_norm, double noise, std::complex<double>** end_tensor)
{
    int ione = 1;
    int szfull = length;

    *tensor = new std::complex<double>[length];
    *end_tensor = new std::complex<double>[length];
    for (int i = 0; i < length; i++) {
        std::complex<double> val (0, 0);
        (*tensor)[i] = val;
        (*end_tensor)[i] = val;
    }

    double phi;
    for(int r = 0; r < R; r++) {
        phi = -100.0 + 200.0 * double(std::rand()) / RAND_MAX;
        for (int i = 0; i < length; i++) {
            std::complex<double> val (cos(phi * i), sin(phi * i));
            (*tensor)[i] += val;
            (*end_tensor)[i] += val;
        }
    }

    right_side_norm = dznrm2_(&szfull, (*tensor), &ione);


    std::random_device rd;
    std::mt19937 gen(rd());
    double a = noise * right_side_norm / sqrt(length);
    std::normal_distribution<double> distribution(0.0, a / sqrt(2)); 

    std::complex<double> sample;

    for (int i = 0; i < length; i++) {
        sample.real(distribution(gen));
        sample.imag(distribution(gen));
        (*tensor)[i] += sample;
    }
}

void
create_matrix(std::complex<double>* &matrix, long long razm, int rank)
{
    matrix = new std::complex<double>[razm * rank];
    for(int r = 0; r < rank; r++) {
        for(int row = 0; row < razm; row++) {
            std::complex<double> val (-1.0 + 2.0 * double(std::rand()) / RAND_MAX, -1.0 + 2.0 * double(std::rand()) / RAND_MAX);
            matrix[row + r * razm] = val;
        }
    }
}


void
create_S_T(std::complex<double>** matrices, int busy_side, int rank, int N, int* razm, std::complex<double>* S, std::complex<double>* res, std::complex<double>* tensor)
{
    

    int* multipliers = new int[N + 1];

    multipliers[N] = 1;
    for(int i = N - 1; i >= 0; i--) {
        multipliers[i] = multipliers[i + 1] * razm[i + 1];
    }

    long long busy_multipliers = multipliers[busy_side];
    for(int i = busy_side; i < N; i++) {
        multipliers[i] = multipliers[i + 1];
    }

    int* mas = new int[N];
    int* mas_razm = new int[N];

    for(int i = 0; i < N + 1; i++) {
        if(i < busy_side) {
            mas_razm[i] = razm[i];
        } else if(i > busy_side) {
            mas_razm[i - 1] = razm[i];
        } else {
            continue;
        }
    }

    long long count = 0, tensor_val = 0;
    std::complex<double> val;

    long long max_r = std::max(rank, razm[busy_side]);

    for(long long r = 0; r < max_r; r++) {

        for(int pas = 0; pas < N; pas++) {
            mas[pas] = 0;
        }

        while(mas[0] < mas_razm[0]) {
            while(mas[N - 1] < mas_razm[N - 1]) {

                if(r < rank) {

                    val = 1;
                    
                    for(int j = 0; j < N + 1; j++) {
                        if(j < busy_side) {
                            val *= matrices[j][r * mas_razm[j] + mas[j]];
                        } else if(j > busy_side) {
                            val *= matrices[j][r * mas_razm[j - 1] + mas[j - 1]];
                        } else {
                            continue;
                        }
                    }

                    S[count] = val;
                }

                if(r < razm[busy_side]) {

                    tensor_val = r * busy_multipliers;

                    for(int i = 0; i < N; i++) {
                        tensor_val += mas[i] * multipliers[i];
                    }

                    res[count] = tensor[tensor_val];
                }

                count++;
                mas[N - 1]++;
            }

            for(int i = N - 1; i >= 1; i--) {
                if(mas[i] >= mas_razm[i]) {
                    mas[i - 1]++;
                    mas[i] = 0;
                }
            }
        }
    }
    delete[] multipliers;
    delete[] mas;
    delete[] mas_razm;
}

double 
ALS(std::complex<double>* tensor, std::complex<double>** matrices, int N, int rank, int* razm, int length, double& right_side_norm)
{
    int szfull = length;
    int ione = 1;
    int info = 654;
    int lwork = 64 * szfull;
    std::complex<double>* work = new std::complex<double>[lwork];
    char cN = 'N';
    int leftsize;
    int leftsize_trunc;

    double relative_residual = 100.0, end_relative_residual = 2.0;
    double diffrent = abs(end_relative_residual - relative_residual);


    std::cout << "Right side norm: " << right_side_norm << std::endl;
    std::complex<double>* S;
    std::complex<double>* T = new std::complex<double>[length];
    int iteration = 0;



    while(diffrent  > 1.0e-11 && iteration < 100000)
    {
        end_relative_residual = relative_residual;
        iteration++;

        for(int i = 0; i < N; i++) {
            S = new std::complex<double>[length / razm[i] * rank];
            // T = new std::complex<double>[length];
            create_S_T(matrices, i, rank, N - 1, razm, S, T, tensor);
            
            leftsize = length / razm[i];
            zgels_(&cN, &leftsize, &rank, &(razm[i]), S, &leftsize, T, &leftsize, work, &lwork, &info);

            relative_residual = 0.0;
            leftsize_trunc = leftsize - rank;

            for(int k = 0; k < razm[i]; k++)
            {
                double tmp = dznrm2_(&leftsize_trunc, T + k * leftsize + rank, &ione);

                relative_residual += tmp * tmp;
            }
            relative_residual = sqrt(relative_residual) / right_side_norm;

            for(int r = 0; r < rank; r++) {
                for(int j = 0; j < razm[i]; j++) {
                    matrices[i][r * razm[i] + j] = T[j * leftsize + r];
                }
            }

            delete[] S;
            // delete[] T;

        }
        diffrent = end_relative_residual - relative_residual;
        if(iteration % 1000 == 0) {
            std::cout << diffrent << std::endl;
        }
        std::cout << "diffrent: " << diffrent << std::endl;
    }

    std::cout << "iteration: " << iteration << std::endl; 

    delete[] work;
    delete[] T;

    return relative_residual;
}







void
create_ALS_tensor(std::complex<double>** matrices, int rank, int N, int* razm, std::complex<double>* tensor, int length, double right_side_norm)
{
    
    std::complex<double>* res = new std::complex<double>[length];

    int* mas = new int[N];

    long long count = 0, tensor_val = 0;
    std::complex<double> val;

    long long max_r = rank;
    std::complex<double> sum;

    for(int i = 0; i < N; i++) {
        mas[i] = 0;
    }

    while(mas[0] < razm[0]) {
        while(mas[N - 1] < razm[N - 1]) {

            res[count] = -tensor[count];

            for(int i = 0; i < rank; i++) {
                sum.real(1);
                sum.imag(0);
                for(int j = 0; j < N; j++) {
                    sum *= matrices[j][mas[j] + i * razm[j]];
                }
                res[count] += sum;
            }

            count++;
            mas[N - 1]++;
        }

        for(int i = N - 1; i >= 1; i--) {
            if(mas[i] >= razm[i]) {
                mas[i - 1]++;
                mas[i] = 0;
            }
        }
    }


    int ione = 1;
    int szfull = length;
    std::cout << "Real error: " << dznrm2_(&szfull, res, &ione) / right_side_norm  << std::endl;

    delete[] res;
    delete[] mas;
}






int
main(void)
{
    std::srand(std::time(nullptr));
    int N, rank, R;
    int length = 1;
    double right_side_norm, noise;
    std::cout << "Введите количество размерностей тензора: ";
    std::cin >> N;

    int* razm = new int[N];
    for(int i = 0; i < N; i++) {
        std::cout << "Введите " << i + 1 << "-ю" << " размерность: ";
        std::cin >> razm[i];
        length *= razm[i];
    }

    std::cout << "Введите желаемый ранг тензора (без учёта шума): ";
    std::cin >> R;

    std::cout << "Введите долю шума: ";
    std::cin >> noise;

    std::cout << "Введите ранг: ";
    std::cin >> rank;

    std::complex<double>* tensor;
    std::complex<double>** matrices = new std::complex<double>*[N];


    std::complex<double>* end_tensor;

    for(int i = 0; i < N; i++) {
        create_matrix(matrices[i], razm[i], rank);
    }

    tensor_make_not_random(&tensor, razm, length, N, R, right_side_norm, noise, &end_tensor);

    std::cout << std::endl << std::endl;

    double relative_residual = ALS(tensor, matrices, N, rank, razm, length, right_side_norm);

    std::cout << "Residual norm: " << relative_residual << std::endl;

    double sum = 0;
    for(int i = 0; i < N; i++) {
        sum += razm[i];
    }

    sum = sqrt((rank * sum) / length);

    std::cout << "Expected error: " << noise * sum << std::endl;

    create_ALS_tensor(matrices, rank, N, razm, end_tensor, length, right_side_norm);

    delete[] razm;
    delete[] tensor;
    delete[] end_tensor;
    for(int i = 0; i < N; i++) {
        delete[] matrices[i];
    }
    delete[] matrices;

    return 0;
}
