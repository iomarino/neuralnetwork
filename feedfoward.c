#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void read_csv(int row, int col, char *filename, double **data){
	FILE *file;
	file = fopen(filename, "r");

	int i = 0;
    char line[4098];
	while (fgets(line, 4098, file) && (i < row))
    {
    	// double row[ssParams->nreal + 1];
        char* tmp = strdup(line);

	    int j = 0;
	    const char* tok;
	    for (tok = strtok(line, ","); tok && *tok; j++, tok = strtok(NULL, ","))
	    {
	        data[i][j] = atof(tok);
	    }
        free(tmp);
        i++;
    }
}
void sample(size_t* v, size_t n, size_t r) {
  if (n>1) {
    for(int i=0; i<r; i++) {
      size_t a = (int)((double)rand()/RAND_MAX*n);
      v[i] = a;
    }
  }
}
void shuffle(size_t* v, size_t n) {
    if (n>1) {
        size_t i;
        for(i=0;i<n-1; i++) {
          size_t j = i+rand()/(RAND_MAX/(n-i)+1);
          int t = v[j];
          v[j] = v[i];
          v[i] = t;
        }
    }
}
double sig(double z) {
	return 1/(1+exp(-z));
}
double dsig(double z) {
	return sig(z)*(1-sig(z));
}
double relu(double z) {
  if(z>0)
    return z;
  else
    return 0;
}
double drelu(double z) {
  if(z>0)
    return 1;
  else
    return 0;
}
double datan(double z) {
  return 1/(1+z*z);
}
double dtanh(double z) {
  return 1/(cosh(z)*cosh(z));
}
double g(double z) { return atan(z); }
double dg(double z) { return datan(z); }

int main() {
  int i, j, k, l, e;
  const int L = 4; // # of layers
  const int K[L] = {784, 256, 128, 10}; // # of neurons per layer
  double ** z = (double **)malloc(L*sizeof(double**));
  for (l=1; l<L; l++) {
    z[l] = (double *) malloc(K[l]*sizeof(double));
  }
  double ** b = (double **)malloc(L*sizeof(double**));
  for (l=1; l<L; l++) {
    b[l] = (double *) malloc(K[l]*sizeof(double));
  }
  double *** w = (double ***)malloc(L*sizeof(double**));
  for (l=1; l<L; l++) {
    w[l] = (double **) malloc(K[l]*sizeof(double *));
    for (k=0; k<K[l]; k++) {
      w[l][k] = (double *)malloc(K[l-1]*sizeof(double));
    }
  }
  double ** a = (double **)malloc(L*sizeof(double**));
  for (l=0; l<L; l++) {
    a[l] = (double *) malloc(K[l]*sizeof(double));
  }
  double ** dCa = (double **)malloc(L*sizeof(double**));
  for (l=1; l<L; l++) {
    dCa[l] = (double *) malloc(K[l]*sizeof(double));
  }
  double ** dCb = (double **)malloc(L*sizeof(double**));
  for (l=1; l<L; l++) {
    dCb[l] = (double *) malloc(K[l]*sizeof(double));
  }
  double *** dCw = (double ***)malloc(L*sizeof(double**));
  for (l=1; l<L; l++) {
    dCw[l] = (double **) malloc(K[l]*sizeof(double *));
    for (k=0; k<K[l]; k++) {
      dCw[l][k] = (double *)malloc(K[l-1]*sizeof(double));
    }
  }
  const double lr = 0.05;
  // generate random weights and biases
  srand(time(0));
  for(l=1; l<L; l++) {
    for(k=0; k<K[l]; k++) {
      b[l][k] = 4*(double)rand()/RAND_MAX-2;
      for(j=0; j<K[l-1]; j++) {
        w[l][k][j] = 2*(double)rand()/RAND_MAX-1;
      }
    }
  }
  // source train inputs and outputs
  unsigned int N = 60000; // dimension of train set
  double ** x = (double **)malloc(N*sizeof(double**));
  for (i=0; i<N; i++) {
    x[i] = (double *)malloc(K[0]*sizeof(double));
  }
  double ** y = (double **)malloc(N*sizeof(double**));
  for (i=0; i<N; i++) {
    y[i] = (double *)malloc(K[L-1]*sizeof(double));
  }
	int row = N;
	int col = K[0]+1;
	double ** data = (double **)malloc(row*sizeof(double *));
	for (int i = 0; i < row; ++i){
		data[i] = (double *)malloc(col*sizeof(double));
	}
	read_csv(row, col, "mnist_train.csv", data);
  for(i=0; i<N; i++) {
    for(j=0; j<K[L-1]; j++) {
      y[i][j] = 0;
      if(data[i][0] == j)
        y[i][j] = 1;
    }
    for(j=0; j<K[0]; j++) {
      x[i][j] = data[i][j+1];
    }
  }

  const unsigned int epochs = 60000;
  const unsigned int sgd_size = 100;
  size_t index[sgd_size];
  // train
  for(e=0; e<epochs; e++) {
		// random sample for SGD
    sample(index, N, sgd_size);
    // reset gradient
    for(l=1; l<L; l++) {
      for(k=0; k<K[l]; k++) {
        dCb[l][k] = 0;
        for(j=0; j<K[l-1]; j++) {
          dCw[l][k][j] = 0;
        }
      }
    }
    // start SGD
    for(i=0; i<sgd_size; i++) {
      // assign activation values for the first layer (input layer)
      for(k=0; k<K[0]; k++) {
        a[0][k] = x[index[i]][k];
      }
      // calculate activation values for the remaining layers
      for(l=1; l<L-1; l++) {
        for(k=0; k<K[l]; k++) {
          z[l][k] = b[l][k];
          for(j=0; j<K[l-1]; j++) {
            z[l][k] += w[l][k][j]*a[l-1][j];
          }
          a[l][k] = g(z[l][k]);
        }
      }
      // exception for output layer
      for(k=0; k<K[L-1]; k++) {
        z[L-1][k] = b[L-1][k];
        for(j=0; j<K[L-2]; j++) {
          z[L-1][k] += w[L-1][k][j]*a[L-2][j];
        }
        a[L-1][k] = sig(z[L-1][k]);
      }
      // calculate derivatives
      for(k=0; k<K[L-1]; k++) {
        dCa[L-1][k] = a[L-1][k]-y[index[i]][k];
        dCb[L-1][k] += dsig(z[L-1][k])*dCa[L-1][k];
        for(j=0; j<K[L-2]; j++) {
          dCw[L-1][k][j] += a[L-2][j]*dsig(z[L-1][k])*dCa[L-1][k];
        }
      }
      for(l=L-2; l>0; l--) {
        for(k=0; k<K[l]; k++) {
          dCa[l][k] = 0;
          for(j=0; j<K[l+1]; j++) {
            dCa[l][k] += w[l+1][j][k]*dg(z[l+1][j])*dCa[l+1][j];
          }
          dCb[l][k] += dg(z[l][k])*dCa[l][k];
          for(j=0; j<K[l-1]; j++) {
            dCw[l][k][j] += a[l-1][j]*dg(z[l][k])*dCa[l][k];
          }
        }
      }
    }
    // gradient descent
    for(l=1; l<L; l++) {
      for(k=0; k<K[l]; k++) {
        b[l][k] += -lr*dCb[l][k]/sgd_size;
        for(j=0; j<K[l-1]; j++) {
          w[l][k][j] += -lr*dCw[l][k][j]/sgd_size;
        }
      }
    }
  }

  // testing
  N = 10000; // dimension of test set
  free(y);
  realloc(x, N*sizeof(double**));
  for (i=0; i<N; i++) {
    x[i] = (double *)malloc(K[0]*sizeof(double));
  }
  // source test inputs and outputs
	row = N;
	col = K[0]+1;
  realloc(data, row*sizeof(double *));
	for (i=0; i<row; i++){
		data[i] = (double *)malloc(col*sizeof(double));
	}
	read_csv(row, col, "mnist_test.csv", data);
  for(i=0; i<N; i++) {
    for(j=0; j<K[0]; j++) {
      x[i][j] = data[i][j+1];
    }
  }
  FILE* fp = fopen("predictions.csv", "w");
  // calculate predictions on test set
  for(i=0; i<N; i++) {
    for(k=0; k<K[0]; k++) {
      a[0][k] = x[i][k];
    }
    for(l=1; l<L-1; l++) {
      for(k=0; k<K[l]; k++) {
        z[l][k] = b[l][k];
        for(j=0; j<K[l-1]; j++) {
          z[l][k] += w[l][k][j]*a[l-1][j];
        }
        a[l][k] = g(z[l][k]);
      }
    }
    for(k=0; k<K[L-1]; k++) {
      z[L-1][k] = b[L-1][k];
      for(j=0; j<K[L-2]; j++) {
        z[L-1][k] += w[L-1][k][j]*a[L-2][j];
      }
      a[L-1][k] = sig(z[L-1][k]);
      fprintf(fp, "%lf,", a[L-1][k]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  // output parameters to file
  fp = fopen("parameters.csv", "w");
  for(l=1; l<L; l++) {
    for(k=0; k<K[l]; k++) {
      fprintf(fp, "%lf\n", b[k][j]);
      for(j=0; j<K[l-1]; j++) {
        fprintf(fp, "%lf ", w[l][k][j]);
      }
      fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
  }
}
