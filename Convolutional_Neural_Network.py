import numpy as np
from numpy import unravel_index

from numpy import genfromtxt
my_data = genfromtxt('train.csv', delimiter=',') # gets first row that is empty of csv
#%%  pictures has 28*28 Dimension 
from random import random
data = np.delete(my_data,0,0) # deleted the get empty row, now there are just datas


train_data = data[0:100]
train_data_y = train_data[:,0]  # data[row][coloumn] or data[row,column]
train_data_features = train_data[:,1:] 


test_data = data[101:111]
test_data_y = test_data[:,0] 
test_data_features = test_data[:,1:] 

#%%

class conv_process():
    
     def __init__(self, input_size,ouput_size,  lr=0.01, epochs=10): 
        
       
        self.output_layer_weights = [{'weights_output_neuron{}'.format(i+1):[random() for i in range(input_size + 1)]} for i in range(ouput_size)]
        # we created output layers' neurons' weights. There are weights as much (input size + 1(bias)) as for each neuron 
            
        self.epochs = epochs
        self.lr = lr 
        
    
     def fit(self, flatten_layer): #
        
        
        for _ in range(self.epochs): 
            
            
            for i in range(1):
                x = np.insert(flatten_layer, 0, 1) 
                x.reshape(len(flatten_layer)+1,1) # without this, dimenison is show (845,) in python
                
                ys = [] # y values
                calc_local_gradients = []
                weight_values = []
                Vs = [] # v values
                
                for i in range(len(self.output_layer_weights)):
                    
                    Vs.append(self.predict(x,self.output_layer_weights[i]) ) # holding "V" values in vs
                
                for i in range(len(self.output_layer_weights)):
                    
                    ys.append(self.transfer_func(Vs[i],Vs)) # holding "y" values in ys



                using_der_softmax = []
                all_derivative_softmaxes = self.derivative_softmax(ys) # take the derivatives according to "ys". ys'lere göre burada türevlerini buluruz
             
                using_der_softmax.append(all_derivative_softmaxes[int(actual_value)]) #The local gradients that will use.
                reshape_using_der_softmax = np.array(using_der_softmax) # 1*10
                
                for i in range(len(self.output_layer_weights)): # calculate local gradients for each output
                    calc_local_gradients.append(self.local_gradients(ys[int(actual_value)],reshape_using_der_softmax[0][i])) 

                
                for i in range(len(self.output_layer_weights)): # this code part updates the weights of output layer and flattens
                    for key, value in dict.items(self.output_layer_weights[i]):
                        weight_values.append(value)
                         
                    w_array_t = np.array(weight_values) # extracts weights in "time t" from dictionary, and writes to this variable
                    
                    self.output_layer_weights[i] = {'weights_of_neuron{}'.format(i+1):np.around(w_array_t[i] + self.lr*calc_local_gradients[i] *x.T,decimals=8)} 
                    #self.output_layer_weights[i] = {'weights_of_neuron{}'.format(i+1):w_array_t[i] + self.lr*calc_local_gradients[i] *x.T} 
                    #deneme.append(self.output_layer_weights[i])
                    
                updated_flattens = self.update_flatten(self.output_layer_weights,calc_local_gradients)
                
            
            
            return updated_flattens
            
        
     def predict(self, x,w): #calculates the V values
         
         ws = []
         for key, value in dict.items(w): # w as dictionary data type. To calculate V values, converts to a list data type
             ws.append(value)
             
         w_array = np.array(ws)
         
         expected_V1 = w_array.dot(x) 
         
         return expected_V1
 
     def transfer_func(self, calculate_V, all_V_Values):# calculates the Y values
         if "e" in str(calculate_V): # codes between line 108 and 116 edits/cuts the values, because when e+ or e- values (exponential values) are too high, np.exp() function  on line 122 shows the resulst as infinity
             asıl_deger = calculate_V
             
             calculate_V = str(calculate_V).replace("["," ")
             n = str(calculate_V).index("e")
             calculate_V = str(calculate_V)
             first_part = calculate_V[:n] 
             asıl_deger = float(first_part)
             calculate_V = asıl_deger
     
         sum_all_exps = 0 # this line implementation of mathematical parts
         for i in range(len(all_V_Values)):
             sum_all_exps += all_V_Values[i]  
         z = np.exp(calculate_V) / sum_all_exps
         return z
     
     def derivative_softmax(self,s):
        matris_array = []
        for i in range(len(s)):
            matris_array.append(float(s[i]))
            
        jacobian_m = np.diag(matris_array) # normalde deneme değil s

        for i in range(len(jacobian_m)): # created a matrix and calculate softmax derivatives
            for j in range(len(jacobian_m)):
                
                if i == j:
                    jacobian_m[i][j] = s[i] * (1 - s[i]) 
                else: 
                    jacobian_m[i][j] = -s[i] * s[j]
        return jacobian_m


     def local_gradients(self,y_correct,derivatives): # calculates the local gradients
        first_part = -1/y_correct 
        second_part = round(derivatives,2)
        local_gradient = first_part * second_part
        return local_gradient
    
    
     def update_flatten(self,output_layer_weights,local_gradients):# updating flatten layer on backward
         flatten_updated = []

         for i in range(flatten_layer.shape[0]):
             summ = 0
             for j in range(len(self.output_layer_weights)):
                 temporal_array = []
                 for k in range(len(self.output_layer_weights)):
                     for key, value in dict.items(self.output_layer_weights[k]):
                         temporal_array.append(value)
                     
                 temporal_array = np.array(temporal_array)
                 
                 summ += temporal_array[j][i]*local_gradients[j] 
                 
             flatten_updated.append(summ)

         return flatten_updated
     
     
     def flatten_to_maxpooling(flatten_layer): # This function assign flatten layer values to max poolings 
         pooling_groups = []
         
         as_numpy = np.array(flatten_layer)
         as_numpy = np.split(as_numpy, 5) # 5 is filter number
         pooling_groups.append(as_numpy)
         
     def filter_updated(self,filter_name,input_normal,gradient_conv):
         
         stride_length = len(gradient_conv)
         h2, w2 = input_normal.shape
         
         backward_convaluated_input = np.empty([3, 3]) # delta_filter
         
         for i in range(h2 - stride_length):
            for j in range(w2 - stride_length):
                region = input_normal[i:(i + stride_length), j:(j + stride_length)]
                #delta_filter = region*gradient_conv
                #delta_filter = np.append (input_normal[i, j], region*gradient_conv)  #region*gradient_conv
                np.append (backward_convaluated_input[i, j], region*gradient_conv)  
                
                #np.append (conv_filter1[i, j], region*gradient_conv)
                    
         filter_name = filter_name + self.lr*backward_convaluated_input#delta filter
         return filter_name
     
     def test_func(self,flatten_layer):
         for i in range(1):
                x = np.insert(flatten_layer, 0, 1) 
                x.reshape(len(flatten_layer)+1,1) 
                
                ys = [] 

                Vs = []
                
                for i in range(len(self.output_layer_weights)):
                    
                    Vs.append(self.predict(x,self.output_layer_weights[i]) ) 
                
                for i in range(len(self.output_layer_weights)):
                    
                    ys.append(self.transfer_func(Vs[i],Vs)) 
         return ys
     


#%%    
if __name__ == '__main__':       
   
    train = []
    for i in range(len(train_data_features)):
        train.append(train_data_features[i].reshape(28,28))
    
    
        # sharpen   0-10 -15-1 0-10 
        # blur  111 111 111
        #Edge enhance 000 -110 000
        #edge detect 010 1-41 010
        #emboss -2-10 -111 012
    sharpen = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    blur = np.array([[1,1,1],[1,1,1],[1,1,1,]])
    edge_enhance = np.array([[0,0,0],[-1,1,0],[0,0,0]])
    edge_detect = np.array([[0, 1, 0],[1,-4,1],[0,1,0]])
    emboss = np.array([[-2, -1, 0],[-1,1,1],[0,1,2]])
    
    num_filters = 5 
        
            # filters is a 3d array with dimensions (num_filters, 3, 3)
            # We divide by 9 to reduce the variance of our initial values
            
    filters = np.random.randn(num_filters, 3, 3) 
    

    for c in range(len(train)): # runs the fit function amount of all train data set
                h, w = train[c].shape
                actual_value = train_data_y[0]
                
                all_conv_filters = []
                
                # conv boyutları 28*28 filterlar 3*3 sonuçta 26*26 olur
                conv_filter1 = np.empty([26, 26])
                conv_filter2 = np.empty([26, 26])
                conv_filter3 = np.empty([26, 26])
                conv_filter4 = np.empty([26, 26])
                conv_filter5 = np.empty([26, 26])
                
                for i in range(h - 2):
                  for j in range(w - 2):
                    im_region = train[c][i:(i + 3), j:(j + 3)]
                    np.append (conv_filter1[i, j], im_region*sharpen)
                    np.append (conv_filter2[i, j], im_region*blur)
                    np.append (conv_filter3[i, j], im_region*edge_enhance)
                    np.append (conv_filter4[i, j], im_region*edge_detect)
                    np.append (conv_filter5[i, j], im_region*emboss)

                
                all_conv_filters.append(conv_filter1)
                all_conv_filters.append(conv_filter2)
                all_conv_filters.append(conv_filter3)
                all_conv_filters.append(conv_filter4)
                all_conv_filters.append(conv_filter5)
                    
                # max pooling 2*2 general dimension => 13*13
                maxpooling_filter1 = np.empty([13, 13])
                maxpooling_filter2 = np.empty([13, 13])
                maxpooling_filter3 = np.empty([13, 13])
                maxpooling_filter4 = np.empty([13, 13])
                maxpooling_filter5 = np.empty([13, 13])
                
                indexes_mpf_one = [] # mpf = max pooling filter
                indexes_mpf_two = []
                indexes_mpf_three = []
                indexes_mpf_four = []
                indexes_mpf_five = []                
                
                for filter_number in range(len(all_conv_filters)):
                    for i in range(h - int(h/2) -1): # range(13)
                      for j in range(w - int(h/2) -1):
                        region = all_conv_filters[filter_number][i:(i + 2), j:(j + 2)] # take parts of convulation layer as 2*2
                        
                        np.append (maxpooling_filter1[i, j], np.amax(region))
                        indexes_mpf_one.append(list(unravel_index(region.argmax(), region.shape)))
                        
                        np.append (maxpooling_filter2[i, j], np.amax(im_region))
                        indexes_mpf_two.append(list(unravel_index(region.argmax(), region.shape)))
                        
                        np.append (maxpooling_filter3[i, j], np.amax(im_region))
                        indexes_mpf_three.append(list(unravel_index(region.argmax(), region.shape)))
                        
                        np.append (maxpooling_filter4[i, j], np.amax(im_region))
                        indexes_mpf_four.append(list(unravel_index(region.argmax(), region.shape)))
                        
                        np.append (maxpooling_filter5[i, j], np.amax(im_region))
                        indexes_mpf_five.append(list(unravel_index(region.argmax(), region.shape)))
    
                flatten_layer_list = []
    
                maxpooling_filter1 = np.nan_to_num(maxpooling_filter1)  # sometimes, there might be NAN values, but we convert NAN values to zero
                maxpooling_filter2 = np.nan_to_num(maxpooling_filter2 )
                maxpooling_filter3 = np.nan_to_num(maxpooling_filter3 )
                maxpooling_filter4 = np.nan_to_num(maxpooling_filter4 )
                maxpooling_filter5 = np.nan_to_num(maxpooling_filter5 )
                
                flatten_layer_list.append(maxpooling_filter1.flatten())
                flatten_layer_list.append(maxpooling_filter2.flatten())
                flatten_layer_list.append(maxpooling_filter3.flatten())
                flatten_layer_list.append(maxpooling_filter4.flatten())
                flatten_layer_list.append(maxpooling_filter5.flatten())
                
                flatten_layer = np.array(flatten_layer_list).flatten() # 13*13 * 5 ==> dimension 845   
                
              
                slf = conv_process(845,10)  

                flat_layers = slf.fit(flatten_layer)
                
                pooling_groups = []
          
                as_numpy = np.array(flat_layers)
                
                as_numpy = np.split(as_numpy, 5) # 5 is filter number , all groups has 169*1 dimension
                
                for i in range(len(as_numpy)): #169*1 dimensions converted to 13*13 
                    as_numpy[i] = as_numpy[i].reshape(13,13)

                
                #updated maxpooling filter arrays added to our real maxpooling filter arrays
                maxpooling_filter1,maxpooling_filter2,maxpooling_filter3,maxpooling_filter4,maxpooling_filter5 = as_numpy

                
                for filter_number in range(len(all_conv_filters)): # convaluations are updated                       
                        count = 0
                        
                        #Conditional statatements  takes index and values according to filter
                        if filter_number == 0: 
                            count = 0
                            indexs = indexes_mpf_one
                        elif filter_number == 1:
                            count = 169
                            indexs = indexes_mpf_two
                        elif filter_number == 2:
                            count = 338
                            indexs = indexes_mpf_three
                        elif filter_number == 3:
                            count = 507
                            indexs = indexes_mpf_four
                        elif filter_number == 4:
                            count = 676
                            indexs = indexes_mpf_five
                            
                        for i in range(13): # 13*13                         
                          for j in range(13):
                            all_conv_filters[filter_number] =  np.zeros([26, 26])                            
                            update_conv = all_conv_filters[filter_number][i:(i + 2), j:(j + 2)]                           
                            update_conv[indexs[count]] = maxpooling_filter1[i, j]                                                        
                            all_conv_filters[filter_number][i:(i + 2), j:(j + 2)] = update_conv[indexs[count]]                                  
                            count += 1
                            
                # filters backward updates
                sharpen = slf.filter_updated(sharpen, train[c], conv_filter1)                        
                blur = slf.filter_updated(blur, train[c], conv_filter2)
                edge_enhance = slf.filter_updated(edge_enhance, train[c], conv_filter3)   
                edge_detect = slf.filter_updated(edge_detect, train[c], conv_filter4)   
                emboss = slf.filter_updated(emboss, train[c], conv_filter5)   
                                                                            
                    
    # after this line , codes in above are adapted to test part, everything is almostly same
    test = []
    for i in range(len(test_data_features)):
        test.append(test_data_features[i].reshape(28,28))
        
    for c in range(len(test)): #range(test_data_size)
                h, w = test[c].shape
                
                all_conv_filters = []
                
                # conv boyutları 28*28 filterlar 3*3 sonuçta 26*26 olur
                conv_filter1 = np.empty([26, 26])
                conv_filter2 = np.empty([26, 26])
                conv_filter3 = np.empty([26, 26])
                conv_filter4 = np.empty([26, 26])
                conv_filter5 = np.empty([26, 26])
                
                for i in range(h - 2):
                  for j in range(w - 2):
                    im_region = test[c][i:(i + 3), j:(j + 3)]
                    np.append (conv_filter1[i, j], im_region*sharpen)
                    np.append (conv_filter2[i, j], im_region*blur)
                    np.append (conv_filter3[i, j], im_region*edge_enhance)
                    np.append (conv_filter4[i, j], im_region*edge_detect)
                    np.append (conv_filter5[i, j], im_region*emboss)
  
                
                all_conv_filters.append(conv_filter1)
                all_conv_filters.append(conv_filter2)
                all_conv_filters.append(conv_filter3)
                all_conv_filters.append(conv_filter4)
                all_conv_filters.append(conv_filter5)
                    
                # max pooling 2*2 general dimension => 13*13
                maxpooling_filter1 = np.empty([13, 13])
                maxpooling_filter2 = np.empty([13, 13])
                maxpooling_filter3 = np.empty([13, 13])
                maxpooling_filter4 = np.empty([13, 13])
                maxpooling_filter5 = np.empty([13, 13])
                
         
                
                for filter_number in range(len(all_conv_filters)):
                    for i in range(h - int(h/2) -1):
                      for j in range(w - int(h/2) -1):
                        im_region = all_conv_filters[filter_number][i:(i + 2), j:(j + 2)]
                        
                        np.append (maxpooling_filter1[i, j], np.amax(im_region))

                        np.append (maxpooling_filter2[i, j], np.amax(im_region))

                        np.append (maxpooling_filter3[i, j], np.amax(im_region))

                        np.append (maxpooling_filter4[i, j], np.amax(im_region))

                        np.append (maxpooling_filter5[i, j], np.amax(im_region))

    
                flatten_layer_list = []
    
                maxpooling_filter1 = np.nan_to_num(maxpooling_filter1) 
                maxpooling_filter2 = np.nan_to_num(maxpooling_filter2 )
                maxpooling_filter3 = np.nan_to_num(maxpooling_filter3 )
                maxpooling_filter4 = np.nan_to_num(maxpooling_filter4 )
                maxpooling_filter5 = np.nan_to_num(maxpooling_filter5 )
                
                flatten_layer_list.append(maxpooling_filter1.flatten())
                flatten_layer_list.append(maxpooling_filter2.flatten())
                flatten_layer_list.append(maxpooling_filter3.flatten())
                flatten_layer_list.append(maxpooling_filter4.flatten())
                flatten_layer_list.append(maxpooling_filter5.flatten())
                
                flatten_layer = np.array(flatten_layer_list).flatten() # 13*13 * 5 ==> dimension is 845   
                #flatten_layer = np.around(flatten_layer,2)
              
                slf = conv_process(845,10)  
                #slf.fit(flatten_layer)   
                print("Predict results : ",slf.test_func(flatten_layer))
                #print(slf.predict(flatten_layer))