# Add APMonitor toolbox available from
# http://apmonitor.com/wiki/index.php/Main/PythonApp
from apm import *
import Tkinter, tkFileDialog

# server and application
s = 'http://byu.apmonitor.com'
a = 'regression'

# clear any prior application
apm(s,a,'clear all')

# load model and data files
apm_load(s,a,'model.apm')
root = Tkinter.Tk()
filename = tkFileDialog.askopenfilename()
csv_load(s,a,filename)

# configure parameters to estimate
apm_info(s,a,'FV','a')
apm_info(s,a,'FV','b')
apm_info(s,a,'FV','c')
apm_option(s,a,'a.status',1)
apm_option(s,a,'b.status',1)
apm_option(s,a,'c.status',1)
apm_option(s,a,'nlc.imode',2)

# solve nonlinear regression
output = apm(s,a,'solve')
print(output)

# retrieve solution
z = apm_sol(s,a)

# print solution
print('Solution')
print('a = ' + str(z['a'][0]))
print('b = ' + str(z['b'][0]))
print('c = ' + str(z['c'][0]))

# plot solution
from matplotlib.pyplot import *
plot(z['xm'],z['ym'],'o',z['xm'],z['y'],'r')
filename_title = os.path.basename(filename)
title(filename_title)
xlabel('Data Entries')
ylabel('PM2.5 Concentration')
legend(['Measured','Predicted'])
xlim(0)
show()
