#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define ANGLE 60.0
#define W_cell 2.0 //セルのガラスの厚さ
#define D_cell 20.0 //セルの直系
#define Diam 14.4 //素子の直径
#define Rad 7.2 //Diam/2.0
#define D_lig 50.0 //中心から光源までの距離
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
double R[3];
double T[3];
double Min_dist;
int Flg_1;
int Flg_2;
int Flg_3;
int Flg_4;
int Flg_5;
int Flg_6;
int Flg_pre;
int Flg_refl;
int Flg_core;
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
	if (dist < Min_dist){
		Min_dist = dist;
	}
	int i;
	if (dist < Rad){
		Flg_core = 1;
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
			R[2] += (D_scr - y)*T[2]/T[1];
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
	double disc;
	double n1;
	double n2;
	double leng;
	double inp;
	double vec_n[3];
	double jud;
	if (k == 3 || k == 4){
		if (k == 3){
			disc = pow((x*T[0] + y*T[1]), 2.0) - (T[0]*T[0] + T[1]*T[1])*(x*x + y*y - Rad*Rad);
			leng = (-(T[0]*x + T[1]*y) - sqrt(disc))/(T[0]*T[0] + T[1]*T[1]);
			n1 = N_oil;
			n2 = NN(Rad, 0.0);
			Flg_3 = 1;
		}else{
			disc = pow((x*T[0] + y*T[1]), 2.0) - (T[0]*T[0] + T[1]*T[1])*(x*x + y*y - Rad*Rad);
			leng = (-(T[0]*x + T[1]*y) + sqrt(disc))/(T[0]*T[0] + T[1]*T[1]);
			n1 = NN(Rad, 0.0);
			n2 = N_oil;
			Flg_4 = 1;
		}
		for (i = 0; i < 3; i++){
			R[i] += leng*T[i];
		}
		vec_n[0] = R[0]/sqrt(R[0]*R[0] + R[1]*R[1]);
		vec_n[1] = R[1]/sqrt(R[0]*R[0] + R[1]*R[1]);
		vec_n[2] = 0.0;
		if (k == 3){
			for (i = 0; i < 3; i++){
				vec_n[i] *= -1.0;
			}
		}
		inp = T[0]*vec_n[0] + T[1]*vec_n[1] + T[2]*vec_n[2];
	}else{
		if (k == 1){
			leng = (Cellbeh_1 - y)/T[1];
			n1 = N_air;
			n2 = N_cell;
			Flg_1 = 1;
		}else if (k == 2){
			leng = (Cellfor_1 - y)/T[1];
			n1 = N_cell;
			n2 = N_oil;
			Flg_2 = 1;
		}else if (k == 5){
			leng = (Cellbeh_2 - y)/T[1];
			n1 = N_oil;
			n2 = N_cell;
			Flg_5 = 1;
		}else{
			leng = (Cellfor_2 - y)/T[1];
			n1 = N_cell;
			n2 = N_air;
			Flg_6 = 1;
		}
		for (i = 0; i < 3; i++){
			R[i] += leng*T[i];
		}
		vec_n[0] = 0.0;
		vec_n[1] = 1.0;
		vec_n[2] = 0.0;
		inp = T[0]*vec_n[0]+T[1]*vec_n[1]+T[2]*vec_n[2];
	}
	jud = n2*n2 - n1*n1 + inp*inp;
	if (jud >= 0.0){
		for (i = 0; i < 3; i++){
			T[i] += (sqrt(jud) - inp)*vec_n[i];
			R[i] += Dt*T[i];
		}
	}else{
		for (i = 0; i < 3; i++){
			T[i] += -2.0*inp*vec_n[i];
			Flg_refl = 1;
		}
	}
}

int main(void){
	int i,j,k;
	char fname1[256];
	char scantmp1[256];
	FILE *fp1;
	char fname2[256];
	char scantmp2[256];
	FILE *fp2;
	double t_abs;
	double ang_pre;
	double ang_now;
	double ang_tmp;
	double ang_det;
	double ang_mar = 0.0;
	int data_num;
	int div_num;
	int div_var;

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

	printf("Input the number of data : ");
	scanf("%d", &data_num);
	printf("Input the number of division : ");
	scanf("%d", &div_num);
	printf("Input the margin of angle : ");
	scanf("%lf", &ang_mar);

	for (i = 0; i < data_num; i++){
		N_hi = 1.5;//1.7 - 0.3*(i/100)/(data_num/100);
		N_dif = 0.015;//0.0003*(1 + i%100);

		double aver = 0.0;
		for (int idx = 1; idx <= 1000; idx++){
			aver += NN(Rad*((double)idx/1000.0), 0.0);
		}
		aver /= 1000.0;//屈折率の平均値

		for (k = 0; k < 8; k++){
			if (k == 0 || k == 4){
				ang_tmp = -ANGLE/2.0;
			}
			if (i%100 == 0 && k == 7){//変更点
					ang_det = ang_tmp;
				}else if(k == 7){
					ang_tmp = ang_det;
				}//ここまで
			div_var = div_num;
			Flg_pre = 0;
			ang_pre = ang_now = ang_tmp;
			if (k < 4){
				Jud = 'p';			
				if (k != 3){
					Flg_pre = 1;
					div_var = pow(10, 2 + k);
				}
			}else{
				Jud = 'n';
				if (k != 7){
					Flg_pre = 1;
					div_var = pow(10, -2 + k);
				}
			}
			for (j = 0; j < div_var; j++){
				if (i%100 != 0 && 3 < k && k < 7){//変更点
					break;
				}//ここまで
				N_oil = NN(Rad, 0.0);
				R[0] = 0.0;
				R[1] = -D_lig;
				R[2] = 0.0;
				ang_tmp = ang_now;
				ang_now = (ang_pre - 2.0*ang_pre*j/(div_var - 1));
				if (ang_mar != 0.0 && (k == 3 || k == 7)){
					ang_now = (ang_pre - ang_mar - 2.0*(ang_pre - ang_mar)*j/(div_var - 1))*M_PI/180.0;
				}
				T[0] = 1.0*tan(ang_now*M_PI/180.0)*N_air;
				T[1] = 1.0*N_air;
				t_abs = sqrt(T[0]*T[0] + T[1]*T[1]);
				T[0] /= t_abs;
				T[1] /= t_abs;
				T[2] = T[0]*sin(M_PI/4.0);
				T[0] = T[0]*cos(M_PI/4.0);
				Flg_1 = 0;
				Flg_2 = 0;
				Flg_3 = 0;
				Flg_4 = 0;
				Flg_5 = 0;
				Flg_6 = 0;
				Flg_refl = 0;
				Flg_core = 0;
				Min_dist = Rad;
				while (R[1] < D_scr){
					Runge(R[0], R[1], R[2]);
					if (Flg_refl == 1){
						printf("Reflection occured\n");
						exit(1);
					}
					if (Flg_core == 1 && Flg_pre == 1){
						break;
					}
				}
				if (Flg_pre == 1){
					if (Flg_core == 1){
						break;
					}else{
						continue;
					}					
				}
				if (j < div_num - 1){
					fprintf(fp1, "%.6lf,", NN(Min_dist, 0.0));
				}else{
					fprintf(fp1, "%.6lf\n", NN(Min_dist, 0.0));
				}	
				//fprintf(fp2, "%f, %f\n", R[0], R[2]);
				if (j < div_num - 1){
					fprintf(fp2, "%.6lf,", R[2]/R[0]);
				}else{
					fprintf(fp2, "%.6lf,%.6lf\n", R[2]/R[0], aver);
				}
			}			
		}
	printf("Data number : %d\n", (i + 1)*2);
	}

	fclose(fp1);
	fclose(fp2);

	printf("Calculation is finished\n");
	return 0;
}