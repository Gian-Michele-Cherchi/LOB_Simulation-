import numpy as np 
import multiprocessing as mp
import sys

######################################################
############## NOISE IMPLEMENTATION ##################
######################################################

sys.path += ["/Users/gmche/Desktop/Internship/llob/fbm-master/"]
from fbm import FBM

###################price calculation function

def price_opt(a,b):
    a_t = np.where(a>0)[0][0]
    b_t = np.where(b>0)[0][-1]
    p_t =(b_t+a_t)/2.
    return p_t,a_t,b_t

#####################################reaction function 

def reaction_opt(a,b):
    react = np.minimum(a,b)
    a -= react
    b -= react
    v = sum(react)
    return a,b,v  
###########################diffusion function

def diffusion_bid(b,orders,prob):
    #How many move? 
    s = np.random.binomial(b,prob)
    #How many to the right(left)? 
    rights = np.random.binomial(s,0.5)
    #Boundary conditions 
    b[0]     += orders[0] - rights[0] + (s[1]-rights[1])
    b[-1]    += rights[-2] -(s[-1]-rights[-1]) 
    #Not boundary
    b[1:N-1] += -s[1:N-1] + rights[:N-2] +(s[2:]-rights[2:])
    return b

def diffusion_ask(a,orders,prob):
    #How many move? 
    z = np.random.binomial(a,prob)
    #How many to the right(left)? 
    leftz  = np.random.binomial(z,0.5)
    #Boundary conditions 
    a[N-1]   += orders[1] - leftz[N-1]  + (z[N-2]-leftz[N-2])
    a[0]     +=  leftz[1] -(z[0]-leftz[0])
    #Not boundary
    a[1:N-1] += -z[1:N-1] + leftz[2:]  +(z[:N-2]- leftz[:N-2])
    return a
def metaorders(a,b,at,bt,q):
            if(q<0):
                i = bt
                q = -q
                while (q>= b[i]):
                    q -= b[i]
                    b[i] = 0 
                    i-= 1
                    if i==0:
                        print('WARNING: Bid boundary reached!')
                        break 
                b[i] -= q
            elif q>0:
                i = at 
                while (q >= a[i]) :
                    q -= a[i]
                    a[i] = 0
                    i += 1
                    if i == len(a):
                        print('WARNING: Ask boundary reached!')
                        break
                a[i] -= q
            return a,b
def rd_w_metaorders(a,b,iterations,orders,noise,rates,prob,return_book):
    pt  = np.zeros(iterations)
    flag = False
    v = 0 
    for t in range(iterations):
        if np.random.binomial(1,0.5) == 1:
            ###############################DIFFUSION  
            b = diffusion_bid(b,orders[t],prob)
            #####################REACTION 
            [a,b,vbid] = reaction_opt(a,b)
            ###############################DIFFUSION  
            a = diffusion_ask(a,orders[t],prob)
            #####################REACTION 
            [a,b,vask] = reaction_opt(a,b)
            v+= vask + vbid
        else: 
            
            ###############################DIFFUSION  
            a = diffusion_ask(a,orders[t],prob)
            #####################REACTION 
            [a,b,vask] = reaction_opt(a,b)
            ###############################DIFFUSION  
            b = diffusion_bid(b,orders[t],prob)
            #####################REACTION 
            [a,b,vbid] = reaction_opt(a,b)
            v+= vask + vbid
        [p_t,a_t,b_t] = price_opt(a,b)
        pt[t]= p_t
        #at[t]= a_t
        #bt[t]= b_t 
        #if return_book == True and t == N_mo:
         #   flag = True
         #   a_book = a[int(at[t]):]
         #   b_book = b[:int(bt[t]+1)]
        #################METAORDERS
        
        #if t < N_mo:
        q_mt = noise[t] + rates[t]
        [a,b] = metaorders(a,b,a_t,b_t,q_mt)
        
    #at -= pt[0]   #Rescaling the price in the reference frame of the order book 
    #bt -= pt[0]
    pt -= pt[0]
    
    if flag == True:
        return pt,v
    else:
        return pt,v

def rd_w_const_rate(a,b,iterations,orders,rates,prob):
   
    pt  = np.zeros(iterations)
    at = np.zeros(iterations)
    bt = np.zeros(iterations)
    
    for t in range(iterations):
        if np.random.binomial(1,0.5) == 1:
            ###############################DIFFUSION  
            b = diffusion_bid(b,orders[t],prob)
            #####################REACTION 
            [a,b] = reaction_opt(a,b)
            ###############################DIFFUSION  
            a = diffusion_ask(a,orders[t],prob)
            #####################REACTION 
            [a,b] = reaction_opt(a,b)
            ##########################PRICE
            
        else: 
            
            ###############################DIFFUSION  
            a = diffusion_ask(a,orders[t],prob)
            #####################REACTION 
            [a,b] = reaction_opt(a,b)
            ###############################DIFFUSION  
            b = diffusion_bid(b,orders[t],prob)
            #####################REACTION 
            [a,b] = reaction_opt(a,b)
            
        [p_t,a_t,b_t] = price_opt(a,b)
        pt[t]= p_t
        at[t]= a_t
        bt[t]= b_t
        
        #################METAORDERS
        #if t < N_mo:
        q_m0 =  rates[t]
        [a,b] = metaorders(a,b,a_t,b_t,q_m0)

#           
        
    at -= pt[0]   #Rescaling the price in the reference frame of the order book 
    bt -= pt[0]
    pt -= pt[0]
    
    return pt
    


#######################################Reaction-Diffusion Process function with parallelization
def getPriceTrajectory(N_prices,current,iterations,dt,prob,m0,return_book,sigma):
    bid = np.zeros(N_prices)
    ask = np.zeros(N_prices)  
    ############################metaorders_initialization + fractional brownian noise 
    fbm = FBM(iterations, 0.75, iterations * dt,method="daviesharte").fgn()
    rates = np.ones(iterations)*m0*dt
    rates = np.floor(rates).astype(int) + np.random.binomial(1,rates-np.floor(rates).astype(int))
    noise = sigma*fbm 
    noise = np.floor(noise).astype(int)
    noise += np.random.binomial(1,sigma*fbm - noise)
    D = prob /(2* dt)
    ll = (current/D)  #Latent Liquidity 
    index = int((N_prices/2)+1)
    ask[index:] = ll * np.arange(index- N_prices/2, N_prices/2,1)
    ask = ask.astype(int) + np.random.binomial(1, ask - ask.astype(int))
    bid[:index-1] = ll * np.arange(N_prices/2, 1, -1)
    bid = bid.astype(int) + np.random.binomial(1, bid - bid.astype(int))
    #orders taken from a poisson distribution of parameter J*dt 
    orders = np.reshape(np.random.poisson(current*dt, size = 2 * iterations), (iterations, 2))
    ######################################computation of participation ratio of m0 wrt all the traded volume in T)
    [pt,v] = rd_w_metaorders(ask,bid,iterations,orders,noise,rates,prob,return_book)
    phi = 1./float(1+float(v/(1+sum(rates)))+float(sigma/(dt*m0))*np.mean(abs(fbm)))
    return [pt,phi]
#index = 1
N =  1001 #number of prices available 
D = 1     #Diffution coefficient
J = 1*D   #Fixed current of orders 
#print('Enter timestep, dt =')
#dt = float(input())
print('Enter sigma/J =')
ratio1 = float(input())
sigma = ratio1*J
#m0_hp = 100*sigma     #high participation ratio rate of metaorders
#print('Enter m0/J =')
#ratio2 = float(input())
#m0_lp = ratio2*J   #low participation ratio ------------------
#m0 = m0_lp     
                           #probability that a trader wants to change the orders' price 
Qtot = (1/2)*(J/D)*(((N-1)/2))**2  #total volume available in the order book at any time 
Q = Qtot/100                      #fixed traded volume of metaorders 
print('Enter Nm=')
#div = float(input())
#Nm = int(Q/(div*m0_lp*dt))
Nm = int(input())
print(Nm)
print('Enter Iterations Nit =')
n = float(input())                                            #Number of steps for the execution of metaorders sucht that T = Nm*dt 
Nit =  int(n*Nm)         #Number of steps of the simulation such that total time is t = Nit*dt 
print('Enter Averages n_av =')
s = int(input())
n_av = s        #number of average 

print('Enter number of points =')
points = int(input())
#price = []
#price_mean = []
#phi_mean = []
res = []

def sim_average(nav):

     
      pool = mp.Pool()
      multiple_results = pool.map_async(func, range(nav))
      results = multiple_results.get()
      pool.terminate()
      #    results = list(map(func, range(nav)))

      #price_mean = np.mean([x[0][0] for x in results], axis = 0)
      #phi_mean = np.mean([x[0][1] for x in results], axis = 1)
      #bid_mean = np.mean([x[0][2] for x in results], axis = 0)
      #ask_mean = np.mean([x[0][1] for x in results], axis = 0) 
      #price_mean0 = np.mean([x[1][0] for x in results], axis = 0) 
      #bid_mean0 = np.mean([x[1][2] for x in results], axis = 0) 
      #ask_mean0 = np.mean([x[1][1] for x in results], axis = 0)
      #len_ask = min([len(x[0][3]) for x in results])
      #len_bid = min([len(x[0][4]) for x in results])
      #ask = np.mean([x[0][3][:len_ask] for x in results], axis = 0) 
      #bid = np.mean([x[0][4][len(x[0][4]) - len_bid:] for x in results], axis = 0)

      return results


m = J*np.linspace(0.001,20,points)
dt = 0.1
for j in range(0,len(m)-1):
    p = 2*D*dt
    def func(i):
        return getPriceTrajectory(N,J,Nit,dt,p,m[j],True,sigma)


    res.append(sim_average(n_av))
    #price_mean.append(res[0])
    #price.append(res[2])
    #phi_mean.append(res[1])

phi = [[x[1] for x in res[i]] for i in range(points-1)]
price = [[x[0] for x in res[i]] for i in range(points-1)]
price_plot = [[x[0][Nit-1] for x in res[i]] for i in range(points-1)]

flag1 = True 
while flag1 == True:

        print("Enter number of averages:")
        n_av = int(input())
        print("Enter number of points m0/J:")
        points = int(input())
        price_final = [[0]]
        k = 0 
        for i in range(len(price_plot)+1):
            if k == len(price_plot) -2: break
            temp = np.concatenate((price_plot[k],price_plot[k+1]),axis = 0)
            k+= 2 
            price_final = np.concatenate((price_final,temp),axis = 0) 
        price_finally = ([x[0] for x in price[1:]])
        phi_final = [[0]]
        k = 0 
        for i in range(len(phi)):
            if k == len(phi)-2: break
            temp = np.concatenate((phi[k],phi[k+1]),axis = 0)
            k+= 2 
            phi_final = np.concatenate((phi_final,temp),axis = 0)
        phi_final = phi_final[1:]
        price_finally = np.log(1+ price_finally)



        var = list(map(lambda i : np.var(price[i],axis = 1), range(0,points-1)))
        var_rescale = [[0]]
        temp = []
        k = 0 
        for i in range(len(var)):
            if k == len(var)-2: break
            temp = np.concatenate((var[k],var[k+1]),axis = 0)
            k+= 2 
            var_rescale = np.concatenate((var_rescale,temp),axis = 0)
        var_rescale = var_rescale[1:]

        ###################################Plot with binning 
        flag2 = True
        while flag2 == True:
        
            print("Enter binsize!")
            binsize = int(input())

            snr = [c/t for c,t in zip(price_finally,np.sqrt(var_rescale))]

            phi_binned = np.array(list(map(lambda i : np.mean(phi[i:i+binsize]), range(0,len(phi)-1-binsize,binsize))))
            norm_price = np.array(list(map(lambda i : np.mean(snr[i:i+binsize]), range(0,len(snr)-1-binsize,binsize))))
            price_binned = np.array(list(map(lambda i : np.mean(price_finally[i:i+binsize]), range(0,len(price_finally)-1-binsize,binsize))))


            f1 = plt.figure(figsize = (15,15))
            plt.title("Co-Impact Curve",fontsize = 20)
            plt.xlabel("$\phi$",fontsize = 20)
            plt.ylabel("$I(\phi)$",fontsize = 20)
            plt.plot(phi_binned,norm_price,'o',color = 'purple')
            plt.grid()
            f1 = plt.savefig("/home/giovanni/files/co_impact/plots/coimpact_norm.jpg")
            f1 = plt.savefig("/home/giovanni/files/co_impact/plots/coimpact_norm.pdf")
            f2 = plt.figure(figsize = (15,15))
            plt.title("Co-Impact Curve",fontsize = 20)
            plt.xlabel("$\phi$",fontsize = 20)
            plt.ylabel("$I(\phi)$",fontsize = 20)
            plt.loglog(phi_binned,norm_price,'o',color = 'purple')
            plt.grid()
            f2 = plt.savefig("/home/giovanni/files/co_impact/plots/loglog_coimpact_norm.jpg")
            f2 = plt.savefig("/home/giovanni/files/co_impact/plots/loglog_coimpact_norm.pdf")

            f3 = plt.figure(figsize = (15,15))
            plt.title("Co-Impact Curve",fontsize = 20)
            plt.xlabel("$\phi$",fontsize = 20)
            plt.ylabel("$I(\phi)$",fontsize = 20)
            plt.plot(phi_binned,price_binned,'o',color = 'purple')
            plt.grid()
            f3 = plt.savefig("/home/giovanni/files/co_impact/co_impact/plots/coimpact.jpg")
            f3 = plt.savefig("/home/giovanni/files/co_impact/plots/coimpact.pdf")
            f4 = plt.figure(figsize = (15,15))
            plt.title("Co-Impact Curve",fontsize = 20)
            plt.xlabel("$\phi$",fontsize = 20)
            plt.ylabel("$I(\phi)$",fontsize = 20)
            plt.loglog(phi_binned,price_binned,'o',color = 'purple')
            plt.grid()
            f4 = plt.savefig("/home/giovanni/files/co_impact/plots/loglog_coimpact.jpg")
            f4 = plt.savefig("/home/giovanni/files/co_impact/plots/loglog_coimpact.pdf")

            print("Do you want to change the binsize? Answer True to do another plot and False to go the next step.")
            flag2 = str(input())



        print("Do you want to do another plot? Answer True to continue and False to stop the execution of the script.")
        flag1 = str(input())














#for i in range(len(price)):
    #f = open("files/co_impact/price"+str(i)+".txt","w+")
    #np.savetxt(f,np.column_stack(price[i][j] for j in range(n_av)))
    #f.close()
#stack_price_plot = np.column_stack(price_plot[i] for i in range(points-1))
#stack_phi = np.column_stack(phi[i] for i in range(points-1))
#f = open("files/co_impact/phi"+str(sigma/J)+".txt","w+")
#np.savetxt(f,stack_phi)
#f.close()
#f = open("files/co_impact/price_plot"+str(sigma/J)+".txt","w+")
#np.savetxt(f,stack_price_plot)
#f.close()


#phi_mean = [np.mean([x[1] for x in res[i]], axis = 0) for i in range(points-1)]
#impact = [np.mean([x[0] for x in res[i]], axis = 0) for i in range(points-1)]
#impact_phi = [impact[i][Nit-1] for i in range(len(impact))]
#price_var = [np.var([x[0] for x in res[i]], axis = 0) for i in range(points-1)]
 

#f = open("files/co_impact/phi_mean"+str(sigma/J)+".txt","w+")
#np.savetxt(f,phi_mean)
#f.close()
#f  = open("files/co_impact/impact_phi"+str(sigma/J)+".txt","w+")
#np.savetxt(f,impact_phi)
#f.close()
#f = open("files/co_impact/impact"+str(sigma/J)+".txt","w+")
#np.savetxt(f,impact)
#f.close()
#f = open("files/co_impact/price_var"+str(sigma/J)+".txt","w+")
#np.savetxt(f,price_var)
#f.close()

##







