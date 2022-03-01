import  numpy as np
import glob
import  os
import matplotlib.pyplot as plt      

def all_npz_to_one(file_format,file_output):
    np_files = glob.glob(file_format)
    print("Processing following files.....")
    for f in np_files:
        print(f)
    all_arrays = {}

    for npfile in np_files:
        array = np.load(npfile)
        for k in array.files:
            if k not in all_arrays.keys():
                all_arrays[k] = array[k]
            else:
                all_arrays[k] = np.concatenate((all_arrays[k],array[k]))

    np.savez_compressed(file_output,**all_arrays)

    
def generate_traj(fish_pos,target_pos,dist_to_path,visualize=False):
    ## will return line_start_positions
    x0 = [fish_pos[0],fish_pos[2]]
    xt= [target_pos[0],target_pos[2]]
    d = dist_to_path
    a = xt[0]-x0[0]
    b = xt[1]-x0[1]
    c = x0[0]**2+x0[1]**2-x0[0]*xt[0]-x0[1]*xt[1]-d*d
    
    p=x0[0]
    q=x0[1]
    r =d 
    hasSOl=True
    
    if np.isclose(b,0):
        k=-c/a
        A=1
        B=-2*q
        C = p*p+q*q+k*k-2*k*p-r*r
        delta = B*B-4*A*C
        if delta<0:
            x = None
            y = None
            hasSOl=False
        elif delta>0:
            y = np.array([(-B+np.sqrt(delta))/(2*A),(-B-np.sqrt(delta))/(2*A)])
            x =np.array([k,k])
        else:
            y = np.array([-B/(2*A)])
            x =np.array([k])
    else:
        m=-a/b
        c=-c/b
        A=m*m+1
        B=2*(m*c-m*q-p)
        C=q*q-r*r+p*p-2*c*q+c*c
        delta = B*B-4*A*C
        if delta<0:
            x = None
            y = None
            hasSOl=False
        elif delta>0:
            x = np.array([(-B+np.sqrt(delta))/(2*A),(-B-np.sqrt(delta))/(2*A)])
            y =m*x+c
        else:
            x = np.array([-B/(2*A)])
            y=m*x+c
            
    if visualize:
        plt.figure()
        plt.scatter(x0[0],x0[1])
        plt.scatter(xt[0],xt[1])
        theta = np.linspace(0, 2 * np.pi, 200)
        plt.plot( d*np.cos(theta)+x0[0],d*np.sin(theta)+x0[1],color="darkred")
        plt.scatter(x,y,color='blue')
        if hasSOl:
            for i in range(x.shape[0]):
                plt.plot([xt[0],x[i]],[xt[1],y[i]],color='green')
        else:
            print("no such lines")
        plt.axis("equal")
        plt.show()
#         if hasSOl:
#             for i in range(x.shape[0]):
#                 print( (x[i]-x0[0])*(x[i]-xt[0])+(y[i]-x0[1])*(y[i]-xt[1]))
    return (hasSOl,np.stack([x,y],axis=1))