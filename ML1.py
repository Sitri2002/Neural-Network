'''
    Author: Jack Nguyen
    Notes: 
    This is a very basic log regression classification
    neural network to predict if someone would survive
    the Titanic based on their age, onboarding family attachments,
    socio-economic status, and sex.
    Being sick right after Thanksgivings is not fun, but
    it gives me enough time to be bored enough to read 50 pages on
    neural networks and watch 12 hours worth of online videos
    on building one. 
'''
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
'''
    Notes on libraries: 
    numpy is used here as a combinational replacement for math and
    random, but you can find any of the library function I used here
    in those libraries if you are not familiar. 
    pandas is used for csv proccesing, but manual scrapping works just
    fine too.
'''
'''  
    Sigmoid activation function, since we are dealing with a classification (binary)
    problem (survived - dead : 1 - 0). Our linear regression model of:
    r = w1*v1 + w2*v2 + ... + wi*vi + b will easily overshoot on continous values,
    we apply it to our sigmoid function to make it stay in [0,1] bounds, and more sensitive to
    uncertain predictions.
'''
def sigmoid(x):
    return 1/(1+np.exp(-x))

'''
    Not entirely sure how to deal with nan values in this dataset (which
    happens a lot in age data), so I just left them out. 
    In a more flexible model I probably can do better than discarding it
    like here, but its linear regression, and setting
    radical value and pray for normalization doesnt seem that great.

    Other than that, fairly simple:

    Socio-economic status is based on ship class: 1st > 2nd > 3rd class
    differ in terms of ticket cost, so we can infer their status through it.
    Sex is binary in data, so we go with male:0 and female:1.
    Onboarding family is an interesting variable, but I conclude that if you
    have family onboard, chances you will stay on board longer to take care
    of them, so it can affect survival chances. This is caluculated by:
    # Family = # Siblings + # Children/Parents/Marriege partners 
'''

def process_data(name):
    df = pd.read_csv(name)
    #  remove nan rows 
    df = df[df['Age'].notna()]
    #  change male/female to 0/1
    df.loc[ df['Sex'] == 'male', 'Sex'] = 0
    df.loc[ df['Sex'] == 'female', 'Sex'] = 1
    return df

df = process_data('docs/train_data_titanic.csv')
validation_df = df.iloc[len(df.index)-101:,:]
test_df = process_data('docs/tes_data_titanic.csv')

def df_to_list(df):
    data = [] 
    for index,rows in df.iterrows():
        row_data = [df['Age'][index], df['Pclass'][index], df['Sex'][index],
        df['Sib'][index] + df['Parch'][index], df['Survived'][index]]
        data.append(row_data)
    return data


data = df_to_list(df)
validation_data = df_to_list(validation_df)

def network():
    ''' 
        Xavier weight initialization for effective sigmoid activation (node = mean(input,output)).
        There are lots of papers and people that can explain this better, but basically
        our original weighting will be changed during gradient descent during each training loop, 
        so a good initialization is important in getting a good model
    '''
    xavier_init = [-(1/np.sqrt((len(df.index)+1)/2)), (1/np.sqrt((len(df.index)+1)/2))]
    w_age = np.random.uniform(xavier_init[0],xavier_init[1])
    w_status = np.random.uniform(xavier_init[0],xavier_init[1])
    w_sex = np.random.uniform(xavier_init[0],xavier_init[1])
    w_family_count = np.random.uniform(xavier_init[0],xavier_init[1])
    b = np.random.uniform(xavier_init[0],xavier_init[1])
    # w_age = np.random.randn()
    # w_status = np.random.randn()
    # w_sex = np.random.randn()
    # w_family_count = np.random.randn()
    # b = np.random.randn()
    ''' 
        Our neural network. This is the place where we train our cleaned data. Remember
        to 'shuffle' your data so that accidental correlated clusters does not
        generate bad outputs!
    '''
    costs = []
    '''
        The rate our network adjust our derivatives, adjust at your own will. Remember,
        too small and it takes more time to run, too big and it might overshoot. Basically
        calculus juggling game.
    '''
    learning_rate = 0.00001
    for i in range(400000):
        ind = np.random.randint(len(data))
        point = data[ind]
        #  linear regression (or line of best fit, just this one for 4 variables and bias)
        r = point[0] * w_age + point[1] * w_status + point[2] * w_sex + point[3] * w_family_count + b
        #  use sigmoid function to get a probability value 
        prediction = sigmoid(r)
        #  target, aka the value we want to predict, this case if they survive the Titanic or not
        target = point[4]
        #  cost mean squared function for current prediction
        cms = np.square(prediction - target)
        #  derivative of cms and prediction
        dcms_dpred = 2 * (prediction - target)
        dpred_dr = sigmoid(r)*(1-sigmoid(r)) 
        #  partial derivatives of best fit line
        dr_dage = point[0]
        dr_dstatus = point[1]
        dr_dsex = point[2]
        dr_dfamily = point[3]
        dr_db = 1
        #  use the derivatives aboves to get partial derivatives of cms by input
        dcms_dage = dcms_dpred * dpred_dr * dr_dage
        dcms_dstatus = dcms_dpred * dpred_dr * dr_dstatus
        dcms_dsex = dcms_dpred * dpred_dr * dr_dsex
        dcms_dfamily = dcms_dpred * dpred_dr * dr_dfamily
        dcms_db = dcms_dpred * dpred_dr * dr_db
        #  going backwards of gradient to least error growth
        w_age -= learning_rate * dcms_dage
        w_status -= learning_rate * dcms_dstatus
        w_sex -= learning_rate * dcms_dsex
        w_family_count -= learning_rate * dcms_dfamily
        b -= learning_rate * dcms_db

        #  error testing 
        if i % 10 == 0:
            cost_sum = 0
            for j in range(len(validation_data)):
                validation_ind = np.random.randint(len(validation_data))
                point = validation_data[validation_ind]
                r = point[0] * w_age + point[1] * w_status + point[2] * w_sex + point[3] * w_family_count + b
                pred = sigmoid(r)
                target = point[4]
                cost_sum += np.square(pred- target)
            costs.append(cost_sum)
    plt.plot(costs/len(validation_data))
    plt.xlabel('# of tests per 100 iteration')
    plt.ylabel('# Mean error squared')
    plt.show()
    return w_age, w_status, w_sex, w_family_count, b

network()