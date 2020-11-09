import numpy as np 
import matplotlib.pyplot as plt
import multiprocessing as mp


######################################################
############## NOISE IMPLEMENTATION ##################
######################################################

def generate_fbm_noise(hurst, sigma, Dt, nb_step):

    covariance = np.zeros((nb_step,nb_step))
    vector = np.arange(1,nb_step+1)**(2*hurst-2)
    covariance[0,1:] += vector[0:nb_step-1]

    for i in range(1,nb_step):
        covariance[i,i+1:] += vector[0:nb_step-1-i]
        covariance[i,0:i] += vector[0:i][::-1]

    covariance *= (2*hurst-1)*hurst*Dt**(2*hurst)
    np.fill_diagonal(covariance, Dt**(2*hurst))
    fbm = np.dot(np.linalg.cholesky(covariance), np.random.normal(0, sigma, nb_step))
    return fbm

def ob_plots(ask,bid,t,N_mo,pt,part_ratio):
    f = plt.figure(figsize=(20,10))
    plt.title('Order Book, $p_t=$'+str(pt),fontsize = 25)
    plt.xlabel('Price',fontsize= 20)
    plt.ylabel('Number of Orders',fontsize = 20)
    if(pt!=0):
        plt.text(x[500], 300,'$ H = 0.75$', fontsize=25)
        plt.text(x[500],250 ,'$\\frac{t}{T}=$'+str(int(abs(t/N_mo))), fontsize=25)
        plt.text(x[500], 175,'$\\frac{m_0}{J}=$'+str(abs(part_ratio)), fontsize=25)
        plt.text(x[500],350 ,'$\\frac{\sigma}{m_{0}}=$'+str(0.1), fontsize=25)
    plt.plot(x[int((N-1)/2)-200:int((N-1)/2)+200],ask[int((N-1)/2)-200:int((N-1)/2)+200],'-',label = '$ask$',color = 'g')
    plt.plot(x[int((N-1)/2)-200:int((N-1)/2)+200],bid[int((N-1)/2)-200:int((N-1)/2)+200],'-',label = '$bid$',color = 'r')
    plt.legend(prop={'size': 20})
    f.savefig('0.75/lowvar/'+str(pt)+'Order_book.jpg') 
    return f 

###################price calculation function

def price_opt(a,b):
    a_t = np.where(a>0)[0][0]
    b_t = np.where(b>0)[0][-1]
    p_t =(b_t+a_t)/2
    return p_t,a_t,b_t

#####################################reaction function 

def reaction_opt(a,b):
    react = np.minimum(a,b)
    a -= react
    b -= react
    return a,b 
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
def rd_w_metaorders(a,b,iterations,orders,mo,prob,N_mo,return_book):
    pt  = np.zeros(iterations)
    at = np.zeros(iterations)
    bt = np.zeros(iterations)
    for t in range(iterations):
        ###############################DIFFUSION  
        b = diffusion_bid(b,orders[t],prob)
        #####################REACTION 
        [a,b] = reaction_opt(a,b)
        ###############################DIFFUSION  
        a = diffusion_ask(a,orders[t],prob)
        #####################REACTION 
        [a,b] = reaction_opt(a,b)
        ##########################PRICE
        [p_t,a_t,b_t] = price_opt(a,b)
        pt[t]= p_t
        at[t]= a_t
        bt[t]= b_t
        #################METAORDERS
	if t <= N_mo:
        	q = mo[t]
        	[a,b] = metaorders(a,b,a_t,b_t,q)
#            if(t==N_mo and j==0 ):
#                f = ob_plots(a,b,t,N_mo,int(p_t -pt[0]),m0[0]/J)
#                plt.show()

    at -= pt[0]   #Rescaling the price in the reference frame of the order book 
    bt -= pt[0]
    pt -= pt[0]
    if return_book == True:
        return pt,at,bt,a,b
    else:
        return pt,at,bt


#######################################Reaction-Diffusion Process function with parallelization
def getPriceTrajectory(N_prices,current,iterations,dt,prob,m0,N_mo,return_book):
    bid = np.zeros(N_prices)
    ask = np.zeros(N_prices)
    ############################metaorders_initialization + fractional brownian noise 
    fbm = generate_fbm_noise(0.75, sigma, dt, N_mo)
    rates = -np.ones(N_mo)*m0*Dt
    rates = np.floor(rates).astype(int) + np.random.binomial(1,rates-np.floor(rates).astype(int))
    noise = fbm
    noise = np.floor(noise).astype(int)
    noise+= np.random.binomial(1,fbm - noise)
    mt = rates + noise
    D = int(prob / dt)
    ll = (current/D)  #Latent Liquidity 
    index = int((N_prices/2)+1)
    ask[index:] = ll * np.arange(index- N_prices/2, N_prices/2,1)
    ask = ask.astype(int) + np.random.binomial(1, ask - ask.astype(int))
    bid[:index-1] = ll * np.arange(N_prices/2, 1, -1)
    bid = bid.astype(int) + np.random.binomial(1, bid - bid.astype(int))
    #orders taken from a poisson distribution of parameter J*dt 
    orders = np.reshape(np.random.poisson(current*dt, size = 2 * iterations), (iterations, 2))

    return rd_w_metaorders(ask,bid,iterations,orders,rates,prob,N_mo,return_book)


N =  1001 #number of prices available 
D = 1     #Diffution coefficient
J = 2*D   #Fixed current of orders 
dt = 0.1  #timestep 
m0_hp = 100*J     #high participation ratio rate of metaorders
m0_lp = 0.1*J    #low participation ratio ------------------
m0 = [m0_hp, m0_lp]      
p = D*dt                           #probability that a trader wants to change the orders' price 
Qtot = (1/2)*(J/D)*(((N-1)/2))**2  #total volume available in the order book at any time 
Q = Qtot/100                      #fixed traded volume of metaorders 
Nm = [int(Q/(m0_hp*dt)), int(Q/(17*m0_lp*dt))] 
                                            #Number of steps for the execution of metaorders sucht that T = Nm*dt 
Nit = [int(2*Nm[0]), int(2*Nm[1])]           #Number of steps of the simulation such that total time is t = Nit*dt 
n_av = 100        #number of average 


index = 1
print(Nit[index])

def func(i):
    return getPriceTrajectory(N,J,Nit[index],dt,p,m0[index],Nm[index],True)

def sim_average(nav):
    
    
    pool = mp.Pool()
    multiple_results = pool.map_async(func, range(nav)) 
    results = multiple_results.get() 
#    results = list(map(func, range(nav)))
    
    price_mean = np.mean([x[0] for x in results], axis = 0) 
    bid_mean = np.mean([x[2] for x in results], axis = 0) 
    ask_mean = np.mean([x[1] for x in results], axis = 0) 
    ask = np.mean([x[3] for x in results], axis = 0) 
    bid = np.mean([x[4] for x in results], axis = 0)
    
    return price_mean, bid_mean, ask_mean,ask,bid,results


res = sim_average(n_av)
price_mean, bid_mean, ask_mean,ask,bid,results = res

price = np.column_stack(results[i][0] for i in range(n_av))
f = open("files/price"+str(m0[index]/J)+".txt","w+")
np.savetxt(f,price)
f.close()

f = open("files/ask"+str(m0[index]/J)+".txt","w+")
np.savetxt(f,ask_mean)
f.close()
f = open("files/bid"+str(m0[index]/J)+".txt","w+")
np.savetxt(f,bid_mean)
f.close()
f = open("files/impact"+str(m0[index]/J)+".txt","w+")
np.savetxt(f,price_mean)
f.close()
f = open("files/book_ask"+str(m0[index]/J)+".txt","w+")
np.savetxt(f,ask)
f.close()
f = open("files/book_bid"+str(m0[index]/J)+".txt","w+")
np.savetxt(f,bid)
f.close()

