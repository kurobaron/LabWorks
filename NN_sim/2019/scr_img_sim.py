# coding: utf-8
import codecs
import time
import numpy as np

#原点(0,0)=測定対象の中心
#光源からスクリーンに向かってy軸を正にとる
#単位はmm

def NN(x, y):
	dist = np.sqrt(x**2.0 + y**2.0)
	if dist <= Rad:
		if Jud == 'p':
			return N_hi*(1.0 - (N_dif/N_hi)/(Rad*Rad)*(x**2.0 + y**2.0))
		else:
			return N_hi*(1.0 - (N_dif/N_hi)/(Rad*Rad)*np.power((Rad - np.abs(np.sqrt(x**2.0 + y**2.0))), 2.0))
	elif Cellfor_1 > y > Cellbeh_1 or Cellfor_2 > y > Cellbeh_2:
			return N_cell
	elif Cellbeh_2 > y > Cellfor_1:
			return N_oil
	else:
			return N_air

def Diffn(x, y, j):
	if j == 0: #屈折率のx方向の偏微分
		if Jud == 'p':
			return -NN(x, y)*2.0*N_dif*x/Rad**2.0
		else:
			return NN(x, y)*2.0*N_dif*x*(Rad/np.sqrt(x**2.0 + y**2.0) - 1.0)/Rad**2.0
	if j == 1: #屈折率のy方向の偏微分
		if Jud == 'p':
			return -NN(x, y)*2.0*N_dif*y/Rad**2.0
		else:
			return NN(x, y)*2.0*N_dif*y*(Rad/np.sqrt(x**2.0 + y**2.0) - 1.0)/Rad**2.0
	if j == 2: #屈折率のz方向の偏微分
		return 0

def Runge(x, y, z):
	global R, T
	dist = np.sqrt(x**2.0 + y**2.0)
	if dist < Rad:
		if Flg_3 == 0:
			Refraction(x, y, 3)
			return
		else:
			aa, bb, cc = np.zeros(3), np.zeros(3), np.zeros(3)
			for i in range(3):
				aa[i] = Dt*Diffn(x, y, i)
			bx = x + Dt*T[0]/2.0 + Dt*aa[0]/8.0
			by = y + Dt*T[1]/2.0 + Dt*aa[1]/8.0
			for i in range(3):
				bb[i] = Dt*Diffn(bx, by, i)
			cx = x + Dt*T[0] + Dt*bb[0]/2.0
			cy = y + Dt*T[1] + Dt*bb[1]/2.0
			for i in range(3):
				cc[i] = Dt*Diffn(cx, cy, i)
				R[i] += Dt*(T[i]+(aa[i]+2.0*bb[i])/6.0)
				T[i] += (aa[i]+4.0*bb[i]+cc[i])/6.0
			return
	else:
		if y > Cellbeh_1 and Flg_1 == 0:
			Refraction(x, y, 1)
			return
		if y > Cellfor_1 and Flg_2 == 0:
			Refraction(x, y, 2)
			return
		if Flg_3 == 1 and Flg_4 ==0:
			Refraction(x, y, 4)
			return
		if y > Cellbeh_2 and Flg_5 == 0:
			Refraction(x, y, 5)
			return
		if y > Cellfor_2 and Flg_6 == 0:
			Refraction(x, y, 6)
			return
		if Flg_6 == 1:
			R[0] += (D_scr - y)*T[0]/T[1]
			R[1] = D_scr
			return
		else:
			for i in range(3):
				R[i] += Dt*T[i]
			return

def Refraction(x, y, k):
	global R, T, Flg_1, Flg_2, Flg_3, Flg_4, Flg_5, Flg_6, Flg_refl
	if k == 3 or k == 4:
		if k == 3:
			disc = (x*T[0] + y*T[1])**2.0 - (T[0]**2.0 + T[1]**2.0)*(x**2.0 + y**2.0 - Rad**2.0)
			dt = (-(T[0]*x + T[1]*y) - np.sqrt(disc))/(T[0]**2.0 + T[1]**2.0)
			n1 = N_oil
			n2 = NN(Rad, 0.0)
			Flg_3 = 1
		if k == 4:
			disc = (x*T[0] + y*T[1])**2.0 - (T[0]**2.0 + T[1]**2.0)*(x**2.0 + y**2.0 - Rad**2.0)
			dt = (-(T[0]*x + T[1]*y) + np.sqrt(disc))/(T[0]**2.0 + T[1]**2.0)
			n1 = NN(Rad, 0.0)
			n2 = N_oil		
			Flg_4 = 1   		
		for i in range(3):
			R[i] += dt*T[i]
		vec_n = R/np.sqrt(R[0]**2 + R[1]**2)
		if R[1] < 0.0:
			vec_n *= -1.0
		inp = T[0]*vec_n[0]+T[1]*vec_n[1]+T[2]*vec_n[2]
	else:
		if k == 1:
			dt = (Cellbeh_1 - y)/T[1]
			n1 = N_air
			n2 = N_cell
			Flg_1 = 1
		if k == 2:
			dt = (Cellfor_1 - y)/T[1]
			n1 = N_cell
			n2 = N_oil
			Flg_2 = 1
		if k == 5:
			dt = (Cellbeh_2 - y)/T[1]
			n1 = N_oil
			n2 = N_cell
			Flg_5 = 1
		if k == 6:
			dt = (Cellfor_2 - y)/T[1]
			n1 = N_cell
			n2 = N_air
			Flg_6 = 1  			
		for i in range(3):
			R[i] += dt*T[i]
		vec_n = np.array([0.0, 1.0, 0.0])
		inp = T[0]*vec_n[0]+T[1]*vec_n[1]+T[2]*vec_n[2]
	jud = n2**2.0 - n1**2.0 + inp**2.0
	if jud >= 0.0:
		for i in range(3):
			T[i] += (np.sqrt(jud) - inp)*vec_n[i]
			R[i] += Dt*T[i]
	else:
		for i in range(3):
			T[i] += -2.0*inp*vec_n[i]
			Flg_refl = 1

if __name__ == '__main__':
	start_time = time.clock()
	Name = input('Input the name of output file : ')
	Fname = Name + '.csv'
	Jud = input('Input GI-POF type (p/n) : ')
	if Jud != 'p' and Jud != 'n':
		print('Please type p or n')
		exit(1)
	N_hi = float(input('Input the highest refractive index : '))
	N_dif = float(input('Input the difference of refractive index : '))
	#Data_num = int(input('Input the number of data : '))
	Div_num = int(input('Input the number of division : '))
	tmp = np.zeros(Div_num)
	"""Mat = np.zeros((2, Data_num, Div_num))
	Mat_nn = np.zeros((Data_num, Div_num))
	Mat_im = np.zeros((Data_num, Div_num))
	for i in range(Data_num):
		N_hi = 1.7 - 0.3*(i//100)/(Data_num//100)
		N_dif = 0.001*(1 + i%100)"""
	for j in range(Div_num):
		W_cell = 2.0 #セルのガラスの厚さ
		D_cell = 20.0 #セルの直系
		Diam = 14.4 #素子の直径
		Rad = Diam/2.0
		D_lig = 15.0 #中心から光源までの距離
		D_scr = 100.0 #中心からスクリーンまでの距離
		N_cell = 1.4708
		N_oil = NN(Rad, 0.0)
		N_air = 1.0
		Cellfor_1 = -D_cell/2.0
		Cellbeh_1 = Cellfor_1 - W_cell
		Cellbeh_2 = D_cell/2.0
		Cellfor_2 = Cellbeh_2 + W_cell
		Dt = 0.005
		R = np.array([-Rad + Diam*j/(Div_num - 1.0), -D_lig, 0.0])
		T = np.array([0.0, 1.0, 0.0])*N_air
		#Mat_nn[i, j] = NN(R[0], 0.0)
		Flg_1 = 0
		Flg_2 = 0
		Flg_3 = 0
		Flg_4 = 0
		Flg_5 = 0
		Flg_6 = 0
		Flg_refl = 0
		while R[1] < D_scr:
			Runge(R[0], R[1], R[2])
			if Flg_refl == 1:
				print('reflection')
				exit(1)
		#Mat_im[i, j] = R[0]
		#print('{}_{} {}'.format(i, j, R[0]))
		print('{} : {}'.format(j, round(R[0], 4)))
		tmp[j] = round(R[0], 4)
	"""Mat[0], Mat[1] = Mat_im, Mat_nn
	np.save(Name, Mat, fix_imports=True)"""
	np.savetxt(Fname, tmp, delimiter = ',') 
	end_time = time.clock()
	print('calculation is finished\n')
	print("TOOK TIME : {0:d} min {1:d} sec".format(int(end_time - start_time) // 60, int((end_time - start_time) % 60)))