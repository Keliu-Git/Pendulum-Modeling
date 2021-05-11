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



def pendulum1 (t, tau):
    return pheta1[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[0]))

def pendulum2 (t, tau):
    return pheta2[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[1]))
def pendulum3 (t, tau):
    return pheta3[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[2]))
def pendulum4 (t, tau):
    return pheta4[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[3]))
def pendulum5 (t, tau):
    return pheta5[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[4]))
def pendulum6 (t, tau):
    return pheta6[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[5]))
def pendulum7 (t, tau):
    return pheta7[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[6]))
def pendulum8 (t, tau):
    return pheta8[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[7]))
def pendulum9 (t, tau):
    return pheta9[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[8]))
def pendulum10 (t, tau):
    return pheta10[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[9]))
def pendulum11 (t, tau):
    return pheta11[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[10]))
def pendulum12 (t, tau):
    return pheta12[0]*np.exp(-t/tau)*np.cos(2*np.pi*(t/period[11]))




############################
popt , pcov = curve_fit(pendulum1, t1, pheta1,  (10), convert_rad(err1), True)
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

popt , pcov = curve_fit(pendulum2, t2, pheta2,  (10), convert_rad(err2), True)
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


popt , pcov = curve_fit(pendulum3, t3, pheta3,  (10), convert_rad(err3), True)
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



popt , pcov = curve_fit(pendulum4, t4, pheta4,  (10), convert_rad(err4), True)
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



popt , pcov = curve_fit(pendulum5, t5, pheta5,  (10), convert_rad(err5), True)
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



popt , pcov = curve_fit(pendulum6, t6, pheta6,  (10), convert_rad(err6), True)
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



popt , pcov = curve_fit(pendulum7, t7, pheta7,  (10), convert_rad(err7), True)
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



popt , pcov = curve_fit(pendulum8, t8, pheta8,  (10), convert_rad(err8), True)
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



popt , pcov = curve_fit(pendulum9, t9, pheta9,  (10), convert_rad(err9), True)
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



popt , pcov = curve_fit(pendulum10, t10, pheta10, (10), convert_rad(err10), True)
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



popt , pcov = curve_fit(pendulum11, t11, pheta11,  (10), convert_rad(err11), True)
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


popt , pcov = curve_fit(pendulum12, t12, pheta12, (10), convert_rad(err12), True)
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
    
[9.36287364]  +-  [0.0791974]
64.08940909094852
[0.5367109]  +-  [0.01234515]
169.0434475573074
[0.45698091]  +-  [0.03791648]
40.47299959672531
[0.46046883]  +-  [0.01944343]
148.43271288731768
[9.63909736]  +-  [0.05718269]
224.05093943780216
[4.53886781]  +-  [0.09859894]
52.94208461145377
[3.43017211]  +-  [0.05402449]
82.40464881696677
[3.79424235]  +-  [0.03518235]
158.63642004451353
[0.08201131]  +-  [inf]
47.932268683117506
[0.17314689]  +-  [0.01227037]
123.49344119269654
[7.42671617]  +-  [0.04600884]
299.04959398653756
[8.8836067]  +-  [0.1219697]
44.37953428656363
'''