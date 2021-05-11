import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t1, x1, y1, pheta1, err1 = np.loadtxt('1 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t2, x2, y2, pheta2, err2 = np.loadtxt('2 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t3, x3, y3, pheta3, err3 = np.loadtxt('3 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t4, x4, y4, pheta4, err4 = np.loadtxt('4 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t5, x5, y5, pheta5, err5 = np.loadtxt('5 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t6, x6, y6, pheta6, err6 = np.loadtxt('6 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t7, x7, y7, pheta7, err7 = np.loadtxt('7 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t8, x8, y8, pheta8, err8 = np.loadtxt('8 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t9, x9, y9, pheta9, err9 = np.loadtxt('9 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t10, x10, y10, pheta10, err10 = np.loadtxt('10 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t11, x11, y11, pheta11, err11 = np.loadtxt('11 Data.csv', delimiter=',', skiprows = 2, unpack = True)
t12, x12, y12, pheta12, err12 = np.loadtxt('12 Data.csv', delimiter=',', skiprows = 2, unpack = True)
trial, com, pheta, mass = np.loadtxt('Data.csv', delimiter=',', skiprows = 1, unpack = True)


def convert_rad (deg):
    return deg*np.pi/180

def T(com):
    return 2*(com)**(1/2)

def chi_square(y_list, f_output, sigma):
    chi_sqr = 0
    for i in range (0, len(y_list)):
        chi_sqr += ((y_list[i] - f_output[i])/sigma[i])**2
    return chi_sqr

pheta1 = convert_rad(pheta1)
pheta2 = convert_rad(pheta2)
pheta3 = convert_rad(pheta3)
pheta4 = convert_rad(pheta4)
pheta5 = convert_rad(pheta5)
pheta6 = convert_rad(pheta6)
pheta7 = convert_rad(pheta7)
pheta8 = convert_rad(pheta8)
pheta9 = convert_rad(pheta9)
pheta10 = convert_rad(pheta10)
pheta11 = convert_rad(pheta11)
pheta12 = convert_rad(pheta12)


com = com/100
pheta = convert_rad(pheta)
period = T(com)



def pendulum1 (t, a, b, c):
    return pheta1[0]*a**(b*t)*np.cos(c*t)

def pendulum2 (t, a, b, c):
    return pheta2[0]*a**(b*t)*np.cos(c*t)

def pendulum3 (t, a, b, c):
    return pheta3[0]*a**(b*t)*np.cos(c*t)

def pendulum4 (t, a, b, c):
    return pheta4[0]*a**(b*t)*np.cos(c*t)

def pendulum5 (t, a, b, c):
    return pheta5[0]*a**(b*t)*np.cos(c*t)

def pendulum6 (t, a, b, c):
    return pheta6[0]*a**(b*t)*np.cos(c*t)

def pendulum7 (t, a, b, c):
    return pheta7[0]*a**(b*t)*np.cos(c*t)

def pendulum8 (t, a, b, c):
    return pheta8[0]*a**(b*t)*np.cos(c*t)

def pendulum9 (t, a, b, c):
    return pheta9[0]*a**(b*t)*np.cos(c*t)

def pendulum10 (t, a, b, c):
    return pheta10[0]*a**(b*t)*np.cos(c*t)

def pendulum11 (t, a, b, c):
    return pheta11[0]*a**(b*t)*np.cos(c*t)

def pendulum12 (t, a, b, c):
    return pheta12[0]*a**(b*t)*np.cos(c*t)



############################
popt , pcov = curve_fit(pendulum1, t1, pheta1,  (1, 1, 1), convert_rad(err1), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t1, pheta1)
plt.plot(t1, pendulum1(t1, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 1')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta1))
for i in range (0, len(pheta1)):
    y_list[i] = pendulum1(t1[i], *popt)
v = len(t1)-len(popt)
chi = chi_square(y_list, pheta1, convert_rad(err1))
print(chi/v)

popt , pcov = curve_fit(pendulum2, t2, pheta2,  (1, 1, 1), convert_rad(err2), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t2, pheta2)
plt.plot(t2, pendulum2(t2, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 2')
print(popt, ' +- ', np.sqrt(pvar))
y_list = np.zeros(len(pheta2))
for i in range (0, len(pheta2)):
    y_list[i] = pendulum2(t2[i], *popt)
v = len(t2)-len(popt)
chi = chi_square(y_list, pheta2, convert_rad(err2))
print(chi/v)


popt , pcov = curve_fit(pendulum3, t3, pheta3,  (1, 1, 1), convert_rad(err3), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t3, pheta3)
plt.plot(t3, pendulum3(t3, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 3')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta3))
for i in range (0, len(pheta3)):
    y_list[i] = pendulum3(t3[i], *popt)
v = len(t3)-len(popt)
chi = chi_square(y_list, pheta3, convert_rad(err3))
print(chi/v)



popt , pcov = curve_fit(pendulum4, t4, pheta4,  (1, 1, 1), convert_rad(err4), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t4, pheta4)
plt.plot(t4, pendulum4(t4, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 4')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta4))
for i in range (0, len(pheta4)):
    y_list[i] = pendulum4(t4[i], *popt)
v = len(t4)-len(popt)
chi = chi_square(y_list, pheta4, convert_rad(err4))
print(chi/v)



popt , pcov = curve_fit(pendulum5, t5, pheta5,  (1, 1, 1), convert_rad(err5), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t5, pheta5)
plt.plot(t5, pendulum5(t5, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 5')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta5))
for i in range (0, len(pheta5)):
    y_list[i] = pendulum5(t5[i], *popt)
v = len(t5)-len(popt)
chi = chi_square(y_list, pheta5, convert_rad(err5))
print(chi/v)



popt , pcov = curve_fit(pendulum6, t6, pheta6,  (1, 1, 1), convert_rad(err6), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t6, pheta6)
plt.plot(t6, pendulum6(t6, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 6')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta6))
for i in range (0, len(pheta6)):
    y_list[i] = pendulum6(t6[i], *popt)
v = len(t6)-len(popt)
chi = chi_square(y_list, pheta6, convert_rad(err6))
print(chi/v)



popt , pcov = curve_fit(pendulum7, t7, pheta7,  (1, 1, 1), convert_rad(err7), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t7, pheta7)
plt.plot(t7, pendulum7(t7, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 7')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta7))
for i in range (0, len(pheta7)):
    y_list[i] = pendulum7(t7[i], *popt)
v = len(t7)-len(popt)
chi = chi_square(y_list, pheta7, convert_rad(err7))
print(chi/v)



popt , pcov = curve_fit(pendulum8, t8, pheta8,  (1, 1, 1), convert_rad(err8), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t8, pheta8)
plt.plot(t8, pendulum8(t8, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 8')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta8))
for i in range (0, len(pheta8)):
    y_list[i] = pendulum8(t8[i], *popt)
v = len(t8)-len(popt)
chi = chi_square(y_list, pheta8, convert_rad(err8))
print(chi/v)



popt , pcov = curve_fit(pendulum9, t9, pheta9,  (1, 1, 1), convert_rad(err9), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t9, pheta9)
plt.plot(t9, pendulum9(t9, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 9')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta9))
for i in range (0, len(pheta9)):
    y_list[i] = pendulum9(t9[i], *popt)
v = len(t9)-len(popt)
chi = chi_square(y_list, pheta9, convert_rad(err9))
print(chi/v)



popt , pcov = curve_fit(pendulum10, t10, pheta10, (1, 1, 1), convert_rad(err10), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t10, pheta10)
plt.plot(t10, pendulum10(t10, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 10')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta10))
for i in range (0, len(pheta10)):
    y_list[i] = pendulum10(t10[i], *popt)
v = len(t10)-len(popt)
chi = chi_square(y_list, pheta10, convert_rad(err10))
print(chi/v)



popt , pcov = curve_fit(pendulum11, t11, pheta11,  (1, 1, 1), convert_rad(err11), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t11, pheta11)
plt.plot(t11, pendulum11(t11, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 11')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta11))
for i in range (0, len(pheta11)):
    y_list[i] = pendulum11(t11[i], *popt)
v = len(t11)-len(popt)
chi = chi_square(y_list, pheta11, convert_rad(err11))
print(chi/v)


popt , pcov = curve_fit(pendulum12, t12, pheta12, (1, 1, 1), convert_rad(err12), True)
pvar = np.diag(pcov)

plt.figure(figsize=(10,10))
plt.scatter(t12, pheta12)
plt.plot(t12, pendulum12(t12, *popt), color='red', marker='|')
plt.xlabel("t (s)")
plt.ylabel("pheta (rad)")
plt.title('Plot 12')
print(popt, ' +- ', np.sqrt(pvar))

y_list = np.zeros(len(pheta12))
for i in range (0, len(pheta12)):
    y_list[i] = pendulum12(t12[i], *popt)
v = len(t12)-len(popt)
chi = chi_square(y_list, pheta12, convert_rad(err12))
print(chi/v)

'''
Output:
[0.9939172  3.79070342 3.7267891 ]  +-  [1.54322660e+02 9.59218387e+04 1.60271531e-04]
9.765348140915622
[0.97655654 2.17157968 3.6575878 ]  +-  [5.55165932e+02 5.20401518e+04 2.08482055e-04]
116.40212523845567
[0.99472977 3.91410776 3.68293051]  +-  [3.14759488e+02 2.32659901e+05 3.23682766e-04]
26.900996000348908
[0.99765831 6.40848977 3.66440535]  +-  [8.44424937e+01 2.27311039e+05 1.75779992e-04]
70.48735515026064
[ 1.00556032 -1.71133591  3.71624646] +-  [6.87864965e+02 2.11844390e+05 1.07563806e-04]
5.53263345170657
[0.99814093 3.15583583 3.69697403]  +-  [4.63430817e+02 7.87423391e+05 2.51341061e-04]
9.281622881826383
[0.99541332 5.24558127 4.47138021]  +-  [1.22475130e+02 1.39811010e+05 2.08966908e-04]
24.743661588107233
[0.99592747 8.3692736  4.46747535]  +-  [4.63222027e+01 9.50805587e+04 1.81887880e-04]
38.88146079131104
[0.97799913 9.86676659 0.83750292]  +-  [3.87301868e+03 1.69266526e+06 1.00391778e-02]
47.97639771121943
[ 0.98040014 11.21074696  0.59903031]  +-  [7.90478288e+02 4.43702011e+05 2.69679443e-03]
117.99912948148916
[0.9977323  3.21200511 4.48952988]  +-  [9.01720641e+01 1.27866339e+05 6.42402243e-05]
7.963336629943003
[  0.99998079 486.39883226   4.50536657]  +-  [1.05200982e-02 2.65107063e+05 1.66651805e-04]
3.370298495931352
'''

