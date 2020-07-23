#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define W_cell 2.0 //セルのガラスの厚さ
#define D_cell 20.0 //セルの直系
#define Diam 14.4 //素子の直径
#define Rad 7.2 //Diam/2.0
#define D_lig 15.0 //中心から光源までの距離
#define D_scr 100.0 //中心からスクリーンまでの距離
#define N_cell 1.4708
#define N_air 1.0
#define Cellfor_1 -10.0 //-D_cell/2.0
#define Cellbeh_1 -12.0 //Cellfor_1 - W_cell
#define Cellbeh_2 10.0 //D_cell/2.0
#define Cellfor_2 12.0 //Cellbeh_2 + W_cell
#define Dt 0.005

double N_hi;
double N_dif;
double N_oil;
double N1;
double N2;
double Leng;
double Inp;
double Vec_n[3];
double R[3];
double T[3];
//double **Mat_nn;
//double **Mat_im;
int Flg_1;
int Flg_2;
int Flg_3;
int Flg_4;
int Flg_5;
int Flg_6;
int Flg_refl;
int Data_num;
int Div_num; 
char Jud;

double NN(double x, double y);
double Diffn(double x, double y, int j);
int Runge(double x, double y, double z);
void Refraction(double x, double y, int k);


double NN(double x, double y){
	double dist = sqrt(x*x + y*y);
	if (dist <= Rad){
		if (Jud == 'p'){
			return N_hi*(1.0 - (N_dif/N_hi)/(Rad*Rad)*(x*x + y*y));
		}else{
			return N_hi*(1.0 - (N_dif/N_hi)/(Rad*Rad)*pow((Rad - fabs(sqrt(x*x + y*y))), 2.0));
		}
	}else if (Cellfor_1 > y > Cellbeh_1 || Cellfor_2 > y > Cellbeh_2){
		return N_cell;
	}else if (Cellbeh_2 > y > Cellfor_1){
		return N_oil;
	}else{
		return N_air;
	}
}

double Diffn(double x, double y, int j){
	if (j == 0){
		if (Jud == 'p'){
			return -NN(x, y)*2.0*N_dif*x/(Rad*Rad);
		}else{
			return NN(x, y)*2.0*N_dif*x*(Rad/sqrt(x*x + y*y) - 1.0)/(Rad*Rad);
		}
	}
	if (j == 1){
		if (Jud == 'p'){
			return -NN(x, y)*2.0*N_dif*y/(Rad*Rad);
		}else{
			return NN(x, y)*2.0*N_dif*y*(Rad/sqrt(x*x + y*y) - 1.0)/(Rad*Rad);
		}
	}
	if (j == 2){
		return 0;
	}
}

int Runge(double x, double y, double z){
	double dist = sqrt(x*x + y*y);
	int i;
	if (dist < Rad){
		if (Flg_3 == 0){
			Refraction(x, y, 3);
			return 0;
		}else{
			double aa[3] = {0};
			double bb[3] = {0};
			double cc[3] = {0};
			for (i = 0; i < 3; i++){
				aa[i] = Dt*Diffn(x, y, i);
			}
			double bx = x + Dt*T[0]/2.0 + Dt*aa[0]/8.0;
			double by = y + Dt*T[1]/2.0 + Dt*aa[1]/8.0;
			for (i = 0; i < 3; i++){
				bb[i] = Dt*Diffn(bx, by, i);
			}
			double cx = x + Dt*T[0] + Dt*bb[0]/2.0;
			double cy = y + Dt*T[1] + Dt*bb[1]/2.0;
			for (i =0; i < 3; i++){
				cc[i] = Dt*Diffn(cx, cy, i);
				R[i] += Dt*(T[i] + (aa[i] + 2.0*bb[i])/6.0);
				T[i] += (aa[i] + 4.0*bb[i] + cc[i])/6.0;
			}
			return 0;
		}
	}else{
		if (y > Cellbeh_1 && Flg_1 == 0){
			Refraction(x, y, 1);
			return 0;
		}else if (y > Cellfor_1 && Flg_2 == 0){
			Refraction(x, y, 2);
			return 0;
		}else if (Flg_3 == 1 && Flg_4 == 0){
			Refraction(x, y, 4);
			return 0;
		}else if (y > Cellbeh_2 && Flg_5 == 0){
			Refraction(x, y, 5);
			return 0;
		}else if (y > Cellfor_2 && Flg_6 == 0){
			Refraction(x, y, 6);
			return 0;
		}else if (Flg_6 == 1){
			R[0] += (D_scr - y)*T[0]/T[1];
			R[1] = D_scr;
			return 0;
		}else{
			for (i = 0; i < 3; i++){
				R[i] += Dt*T[i];
			}
			return 0;
		}
	}
}

void Refraction(double x, double y, int k){
	int i;
	if (k == 3 || k == 4){
		if (k == 3){
			double disc = pow((x*T[0] + y*T[1]), 2.0) - (T[0]*T[0] + T[1]*T[1])*(x*x + y*y - Rad*Rad);
			Leng = (-(T[0]*x + T[1]*y) - sqrt(disc))/(T[0]*T[0] + T[1]*T[1]);
			N1 = N_oil;
			N2 = NN(Rad, 0.0);
			Flg_3 = 1;
		}else{
			double disc = pow((x*T[0] + y*T[1]), 2.0) - (T[0]*T[0] + T[1]*T[1])*(x*x + y*y - Rad*Rad);
			Leng = (-(T[0]*x + T[1]*y) + sqrt(disc))/(T[0]*T[0] + T[1]*T[1]);
			N1 = NN(Rad, 0.0);
			N2 = N_oil;
			Flg_4 = 1;
		}
		for (i = 0; i < 3; i++){
			R[i] += Leng*T[i];
		}
		Vec_n[0] = R[0]/sqrt(R[0]*R[0] + R[1]*R[1]);
		Vec_n[1] = R[1]/sqrt(R[0]*R[0] + R[1]*R[1]);
		Vec_n[2] = 0.0;
		if (R[1] < 0.0){
			for (i = 0; i < 3; i++){
				Vec_n[i] *= -1.0;
			}
		}
		Inp = T[0]*Vec_n[0]+T[1]*Vec_n[1]+T[2]*Vec_n[2];
	}else{
		if (k == 1){
			Leng = (Cellbeh_1 - y)/T[1];
			N1 = N_air;
			N2 = N_cell;
			Flg_1 = 1;
		}else if (k == 2){
			Leng = (Cellfor_1 - y)/T[1];
			N1 = N_cell;
			N2 = N_oil;
			Flg_2 = 1;
		}else if (k == 5){
			Leng = (Cellbeh_2 - y)/T[1];
			N1 = N_oil;
			N2 = N_cell;
			Flg_5 = 1;
		}else{
			Leng = (Cellfor_2 - y)/T[1];
			N1 = N_cell;
			N2 = N_air;
			Flg_6 = 1;
		}
		for (i = 0; i < 3; i++){
			R[i] += Leng*T[i];
		}
		Vec_n[0] = 0.0;
		Vec_n[1] = 1.0;
		Vec_n[2] = 0.0;
		Inp = T[0]*Vec_n[0]+T[1]*Vec_n[1]+T[2]*Vec_n[2];
	}
	double jud = N2*N2 - N1*N1 + Inp*Inp;
	if (jud >= 0.0){
		for (i = 0; i < 3; i++){
			T[i] += (sqrt(jud) - Inp)*Vec_n[i];
			R[i] += Dt*T[i];
		}
	}else{
		for (i = 0; i < 3; i++){
			T[i] += -2.0*Inp*Vec_n[i];
			Flg_refl = 1;
		}
	}
}

/*double Inpdata(char *disp, double a){
	double drtn;		
	char scantmp[256]; 
	char *endptr;
	printf("%s (%10.5Lf) = ", disp, a);
	fgets(scantmp, 256, stdin); 
	drtn = strtod(scantmp, &endptr);
	if (scantmp == endptr){
		drtn=a;
	}
	return drtn;
}*/

int main(void){
    int i, j, k;
	char fname1[256];
	char scantmp1[256];
	FILE *fp1;
	char fname2[256];
	char scantmp2[256];
	FILE *fp2;

	strcpy(fname1, "");
	printf("Input the name of file for refractive index : ");
	fgets(scantmp1, 256, stdin); 
	if(strlen(scantmp1) == 1){
		fp1 = stdout;
	}else{
		strncat(fname1, scantmp1, strlen(scantmp1) - 1);
		strncat(fname1, ".csv", 4);
		printf("The name of file for refractive index = %s \n", fname1);
		if ((fp1 = fopen(fname1, "w")) == NULL){
			printf("Can't create the outputfile \"%s\" \n", fname1);
			exit(1);
		}
		}

	strcpy(fname2, "");
	printf("Input the name of file for screen image : ");
	fgets(scantmp2, 256, stdin); 
	if(strlen(scantmp2) == 1){
		fp2 = stdout;
	}else{
		strncat(fname2, scantmp2, strlen(scantmp2) - 1);
		strncat(fname2, ".csv", 4);
		printf("The name of file for screen image = %s \n", fname2);
		if ((fp2 = fopen(fname2, "w")) == NULL){
			printf("Can't create the output file \"%s\" \n", fname2);
			exit(1);
		}
		}

	/*printf("Input GI-POF type (p/n) : ");
	Jud = getc(stdin);
	if 	(Jud != 'p' && Jud != 'n'){
		printf("Please type p or n\n");
		exit(1);
	}*/

	printf("Input the number of data : ");
	scanf("%d", &Data_num);
	printf("Input the number of division : ");
	scanf("%d", &Div_num);

/*	Mat_nn = (double **)malloc(Data_num*sizeof(double *));
	Mat_im = (double **)malloc(Data_num*sizeof(double *));
	for (i = 0; i < Data_num; i++){
		Mat_nn[i] = (double *)malloc(Div_num*sizeof(double));
		Mat_im[i] = (double *)malloc(Div_num*sizeof(double));
	}	*/

	for (i = 0; i < Data_num; i++){
		N_hi = 1.7 - 0.3*(i/100)/(Data_num/100);
		N_dif = 0.0003*(1 + i%100); //屈折率差0.0003~0.03
		for (k = 0; k < 2; k++){
			if (k == 0){
				Jud = 'p';
			}else{
				Jud = 'n';
			}
			for (j = 0; j < Div_num; j++){
				N_oil = NN(Rad, 0.0);
				R[0] = -Rad + Diam*j/(Div_num - 1.0);
				R[1] = -D_lig;
				R[2] = 0.0;
				T[0] = 0.0*N_air;
				T[1] = 1.0*N_air;
				T[2] = 0.0*N_air;
				//Mat_nn[i][j] = NN(R[0], 0.0);
				if (j < Div_num - 1){
					fprintf(fp1, "%lf,", round(NN(R[0], 0.0)*1000000)/1000000); //小数第6位以上はPythonとの誤差を無視できるとした
				}else{
					fprintf(fp1, "%lf\n", round(NN(R[0], 0.0)*1000000)/1000000);
				}		
				Flg_1 = 0;
				Flg_2 = 0;
				Flg_3 = 0;
				Flg_4 = 0;
				Flg_5 = 0;
				Flg_6 = 0;
				Flg_refl = 0;
				while (R[1] < D_scr){
					Runge(R[0], R[1], R[2]);
					if (Flg_refl == 1){
						printf("Reflection occured\n");
						exit(1);
					}
				}
				//Mat_im[i][j] = R[0];
				if (j < Div_num - 1){
					fprintf(fp2, "%lf,", round(R[0]*1000000)/1000000); //小数第6位以上はPythonとの誤差を無視できるとした
				}else{
					fprintf(fp2, "%lf\n", round(R[0]*1000000)/1000000);
				}
			}			
		}
	printf("Data number : %d\n", (i + 1)*2);
	}

	fclose(fp1);
	fclose(fp2);
	//free(Mat_nn);
	//free(Mat_im);

	printf("Calculation is finished\n");
	return 0;
	}
