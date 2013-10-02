#!/usr/bin/env python
from string import *
import sys
import struct
import array

#script for generating the damping matrix used in the LSQR inversion

#input parameters:
#argv[1]: prefix of the output file name
#argv[2]: starting row number for the damping matrix
#argv[3]: starting col number for the damping matrix
#argv[4]: grid number in x direction
#argv[5]: grid number in y direction
#argv[6]: grid number in z direction
#argv[7]: number of parameters
#argv[8]: identity damping weight
#argv[9]: Laplacian damping weight (xx)
#argv[10]: Laplacian damping weight (yy)
#argv[11]: Laplacian damping weight (zz)
#argv[12]: Laplacian damping weight (xy)
#argv[13]: Laplacian damping weight (xz)
#argv[14]: Laplacian damping weight (yz)

#output files:
#(1) prefix_row_info.bin : number of non-zero per row (int)
#(2) prefix_row_data.bin : row-based row,col,value (int,int,double)
#(3) prefix_col_info.bin : number of non-zero per col (int)  
#(4) prefix_col_data.bin : rcol-based row,col,value (int,int,double) 


d0=[]
#======= read inputs
row0=int(sys.argv[1])
col0=int(sys.argv[2]) 
nx=int(sys.argv[4])
ny=int(sys.argv[5])
nz=int(sys.argv[6])
np=int(sys.argv[7])
d0.append(float(sys.argv[8]))
d0.append(float(sys.argv[9]))
d0.append(float(sys.argv[10]))
d0.append(float(sys.argv[11]))
d0.append(float(sys.argv[12]))
d0.append(float(sys.argv[13]))
d0.append(float(sys.argv[14]))

#======= calculate number of non-zero
print ('Before generating array for storing damping!')
total_row=nx*ny*nz*np+(nx-2)*ny*nz*np+nx*(ny-2)*nz*np+nx*ny*(nz-2)*np+(nx-2)*(ny-2)*nz*np+(nx-2)*ny*(nz-2)*np+nx*(ny-2)*(nz-2)*np
print ('total row :'+str(int(total_row))+'\n')

#====== function for calaulating col
def idmap(iix,iiy,iiz):
    #print nx, ny, nz
    #input('?')
    return iiz*ny*nx+iiy*nx+iix+col0

#====== only store np=1 and then copy to np=2                                                                                   
np1=int(1)
nnz_number=nx*ny*nz*np1+(nx-2)*ny*nz*np1*3+nx*(ny-2)*nz*np1*3+nx*ny*(nz-2)*np1*3+(nx-2)*(ny-2)*nz*np1*4+(nx-2)*ny*(nz-2)*np1*4+nx*(ny-2)*(nz-2)*np1*4

col_idx=array.array('i',(0 for i in range(0,nnz_number))) #creater 'i' integer array                                                                                     
print ('Col idx OK!')
col_sort_idx=array.array('i',(0 for i in range(0,nnz_number))) #creater 'i' integer array                                   
print ('sort Col idx OK!')
row_idx=array.array('i',(0 for i in range(0,nnz_number))) #creater 'i' integer array           
print ('Row idx OK!')
#damp_value=array.array('d',(0 for i in range(0,nnz_number))) #creater 'd' 64-bit float array                                          
damp_value=array.array('f',(0 for i in range(0,nnz_number))) #creater 'f' 32-bit float array           
print ('value vector OK!')


#******** start row 
row=row0
nnz_counter=int(0);
#fout1=open(sys.argv[3]+'_row_data','w')
#fout3=open(sys.argv[3]+'_row_info','w')
#fout2=open(sys.argv[3]+'_col_index','w')
row_info_bin=open(sys.argv[3]+'_row_info.bin','wb')
row_data_bin=open(sys.argv[3]+'_row_data.bin','wb')
print 'constructing identity damping, starting from row...', row0

for ip in range(np):
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
		#=========== check ==========
		if(((int(iz)%2)==0)&(int(ix)==0)&(int(iy)==0)):
                    #print("working on: "+str(row)+'/'+str(total_row)+'\n')
                    print("working on: "+str(row)+'/'+str(total_row)+' = '+str(float(float(row)/total_row)*100.0)+'%\n')
		#=========================
                col=idmap(ix,iy,iz)
                col=col+ip*nx*ny*nz
                val=1.0*d0[0]
		#====== write binary file
		info=struct.pack("i",1)  # "i" signed interger
		row_info_bin.write(info)
                #fout3.write('1\n')

		data=struct.pack("2id",row,col,val) # 2 i and "d" 64-bit float
                #data=struct.pack("2if",row,col,val) # 2 i and "d" 32-bit float
		row_data_bin.write(data)
		#===== store value
		if(ip==0):
                    #fout1.write(str(int(row))+' '+str(int(col))+' '+str(val)+'\n')
                    #fout2.write(str(int(col))+'\n')
                    row_idx[nnz_counter]=row
                    col_idx[nnz_counter]=col
                    damp_value[nnz_counter]=val
                    nnz_counter=nnz_counter+1

                #fout1.write('1 '+str(col)+' '+str(val)+'\n')
		#fout1.write(str(col)+' 1 '+str(val)+'\n')
                row=row+1
		#===== for xx
		if(ix<(nx-2)):
			#====== write binary file
			info=struct.pack("i",3)
			row_info_bin.write(info)
                        #fout3.write('3\n')

			col1=idmap(ix,iy,iz)
	                col1=col1+ip*nx*ny*nz
	                val1=1.0*d0[1]
			#====== write binary file
			data=struct.pack("2id",row,col1,val1)
                        #data=struct.pack("2if",row,col1,val1)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col1
                            damp_value[nnz_counter]=val1
                            nnz_counter=nnz_counter+1
                            #fout1.write(str(int(row))+' '+str(int(col1))+' '+str(val1)+'\n')
                            #fout2.write(str(int(col1))+'\n')

			col2=idmap(ix+1,iy,iz)
                	col2=col2+ip*nx*ny*nz
                	val2=-2.0*d0[1]
			#====== write binary file
			data=struct.pack("2id",row,col2,val2)
                        #data=struct.pack("2if",row,col2,val2)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col2))+' '+str(val2)+'\n')
                            #fout2.write(str(int(col2))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col2
                            damp_value[nnz_counter]=val2
                            nnz_counter=nnz_counter+1

			col3=idmap(ix+2,iy,iz)
                	col3=col3+ip*nx*ny*nz
                	val3=1.0*d0[1]
			#====== write binary file
			data=struct.pack("2id",row,col3,val3)
                        #data=struct.pack("2if",row,col3,val3)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col3))+' '+str(val3)+'\n')
                            #fout2.write(str(int(col3))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col3
                            damp_value[nnz_counter]=val3
                            nnz_counter=nnz_counter+1

			#fout1.write('3 '+str(col1)+' '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+'\n')
			#fout1.write(str(col1)+' 2 '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+'\n')
			row=row+1
		#===== for yy
		if(iy<(ny-2)):
			#====== write binary file
			info=struct.pack("i",3)
			row_info_bin.write(info)
                        #fout3.write('3\n')

			col1=idmap(ix,iy,iz)
	                col1=col1+ip*nx*ny*nz
	                val1=1.0*d0[2]
			#====== write binary file
			data=struct.pack("2id",row,col1,val1)
                        #data=struct.pack("2if",row,col1,val1)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col1))+' '+str(val1)+'\n')
                            #fout2.write(str(int(col1))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col1
                            damp_value[nnz_counter]=val1
                            nnz_counter=nnz_counter+1

			col2=idmap(ix,iy+1,iz)
                	col2=col2+ip*nx*ny*nz
                	val2=-2.0*d0[2]
			#====== write binary file
			data=struct.pack("2id",row,col2,val2)
                        #data=struct.pack("2if",row,col2,val2)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col2))+' '+str(val2)+'\n')
                            #fout2.write(str(int(col2))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col2
                            damp_value[nnz_counter]=val2
                            nnz_counter=nnz_counter+1

			col3=idmap(ix,iy+2,iz)
                	col3=col3+ip*nx*ny*nz
                	val3=1.0*d0[2]
			#====== write binary file
			data=struct.pack("2id",row,col3,val3)
                        #data=struct.pack("2if",row,col3,val3)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col3))+' '+str(val3)+'\n')
                            #fout2.write(str(int(col3))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col3
                            damp_value[nnz_counter]=val3
                            nnz_counter=nnz_counter+1

			#fout1.write('3 '+str(col1)+' '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+'\n')
			#fout1.write(str(col1)+' 3 '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+'\n')
			row=row+1
		#===== for zz
		if(iz<(nz-2)):
			#====== write binary file
			info=struct.pack("i",3)
			row_info_bin.write(info)
                        #fout3.write('3\n')

			col1=idmap(ix,iy,iz)
	                col1=col1+ip*nx*ny*nz
	                val1=1.0*d0[3]
			#====== write binary file
			data=struct.pack("2id",row,col1,val1)
                        #data=struct.pack("2if",row,col1,val1)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col1))+' '+str(val1)+'\n')
                            #fout2.write(str(int(col1))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col1
                            damp_value[nnz_counter]=val1
                            nnz_counter=nnz_counter+1

			col2=idmap(ix,iy,iz+1)
                	col2=col2+ip*nx*ny*nz
                	val2=-2.0*d0[3]
			#====== write binary file
			data=struct.pack("2id",row,col2,val2)
                        #data=struct.pack("2if",row,col2,val2)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col2))+' '+str(val2)+'\n')
                            #fout2.write(str(int(col2))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col2
                            damp_value[nnz_counter]=val2
                            nnz_counter=nnz_counter+1

			col3=idmap(ix,iy,iz+2)
                	col3=col3+ip*nx*ny*nz
                	val3=1.0*d0[3]
			#====== write binary file
			data=struct.pack("2id",row,col3,val3)
                        #data=struct.pack("2if",row,col3,val3)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col3))+' '+str(val3)+'\n')
                            #fout2.write(str(int(col3))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col3
                            damp_value[nnz_counter]=val3
                            nnz_counter=nnz_counter+1

			#fout1.write('3 '+str(col1)+' '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+'\n')
			#fout1.write(str(col1)+' 4 '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+'\n')
			row=row+1
		#====== for xy 
		if((ix<(nx-2))&(iy<(ny-2))):
			#====== write binary file
			info=struct.pack("i",4)
			row_info_bin.write(info)
                        #fout3.write('4\n')

			col1=idmap(ix,iy,iz)
                	col1=col1+ip*nx*ny*nz
                	val1=0.5*d0[4]
                	#====== write binary file
			data=struct.pack("2id",row,col1,val1)
                        #data=struct.pack("2if",row,col1,val1)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col1))+' '+str(val1)+'\n')
                            #fout2.write(str(int(col1))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col1
                            damp_value[nnz_counter]=val1
                            nnz_counter=nnz_counter+1

                	col2=idmap(ix+2,iy,iz)
                	col2=col2+ip*nx*ny*nz
                	val2=-0.5*d0[4]
			#====== write binary file
			data=struct.pack("2id",row,col2,val2)
                        #data=struct.pack("2if",row,col2,val2)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col2))+' '+str(val2)+'\n')
                            #fout2.write(str(int(col2))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col2
                            damp_value[nnz_counter]=val2
                            nnz_counter=nnz_counter+1

                	col3=idmap(ix,iy+2,iz)
                	col3=col3+ip*nx*ny*nz
                	val3=-0.5*d0[4]
			#====== write binary file
			data=struct.pack("2id",row,col3,val3)
                        #data=struct.pack("2if",row,col3,val3)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col3))+' '+str(val3)+'\n')
                            #fout2.write(str(int(col3))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col3
                            damp_value[nnz_counter]=val3
                            nnz_counter=nnz_counter+1

                	col4=idmap(ix+2,iy+2,iz)
                	col4=col4+ip*nx*ny*nz
                	val4=0.5*d0[4]
			#====== write binary file
			data=struct.pack("2id",row,col4,val4)
                        #data=struct.pack("2if",row,col4,val4)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col4))+' '+str(val4)+'\n')
                            #fout2.write(str(int(col4))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col4
                            damp_value[nnz_counter]=val4
                            nnz_counter=nnz_counter+1

                	#fout1.write('4 '+str(col1)+' '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+' '+str(col4)+' '+str(val4)+'\n')
			#fout1.write(str(col1)+' 5 '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+' '+str(col4)+' '+str(val4)+'\n')
                	row=row+1
		#====== for xz 
		if((ix<(nx-2))&(iz<(nz-2))):
			#====== write binary file
			info=struct.pack("i",4)
			row_info_bin.write(info)
                        #fout3.write('4\n')

			col1=idmap(ix,iy,iz)
                	col1=col1+ip*nx*ny*nz
                	val1=0.5*d0[5]
                	#====== write binary file
			data=struct.pack("2id",row,col1,val1)
                        #data=struct.pack("2if",row,col1,val1)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col1))+' '+str(val1)+'\n')
                            #fout2.write(str(int(col1))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col1
                            damp_value[nnz_counter]=val1
                            nnz_counter=nnz_counter+1

                	col2=idmap(ix+2,iy,iz)
                	col2=col2+ip*nx*ny*nz
                	val2=-0.5*d0[5]
			#====== write binary file
			data=struct.pack("2id",row,col2,val2)
                        #data=struct.pack("2if",row,col2,val2)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col2))+' '+str(val2)+'\n')
                            #fout2.write(str(int(col2))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col2
                            damp_value[nnz_counter]=val2
                            nnz_counter=nnz_counter+1

                	col3=idmap(ix,iy,iz+2)
                	col3=col3+ip*nx*ny*nz
                	val3=-0.5*d0[5]
			#====== write binary file
			data=struct.pack("2id",row,col3,val3)
                        #data=struct.pack("2if",row,col3,val3)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col3))+' '+str(val3)+'\n')
                            #fout2.write(str(int(col3))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col3
                            damp_value[nnz_counter]=val3
                            nnz_counter=nnz_counter+1

                	col4=idmap(ix+2,iy,iz+2)
                	col4=col4+ip*nx*ny*nz
                	val4=0.5*d0[5]
			#====== write binary file
			data=struct.pack("2id",row,col4,val4)
                        #data=struct.pack("2if",row,col4,val4)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):	
                            #fout1.write(str(int(row))+' '+str(int(col4))+' '+str(val4)+'\n')
                            #fout2.write(str(int(col4))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col4
                            damp_value[nnz_counter]=val4
                            nnz_counter=nnz_counter+1

                	#fout1.write('4 '+str(col1)+' '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+' '+str(col4)+' '+str(val4)+'\n')
			#fout1.write(str(col1)+' 6 '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+' '+str(col4)+' '+str(val4)+'\n')
                	row=row+1
		#====== for yz 
		if((iy<(ny-2))&(iz<(nz-2))):
			#====== write binary file
			info=struct.pack("i",4)
			row_info_bin.write(info)
                        #fout3.write('4\n')

			col1=idmap(ix,iy,iz)
                	col1=col1+ip*nx*ny*nz
                	val1=0.5*d0[6]
                	#====== write binary file
			data=struct.pack("2id",row,col1,val1)
                        #data=struct.pack("2if",row,col1,val1)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col1))+' '+str(val1)+'\n')
                            #fout2.write(str(int(col1))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col1
                            damp_value[nnz_counter]=val1
                            nnz_counter=nnz_counter+1

                	col2=idmap(ix,iy+2,iz)
                	col2=col2+ip*nx*ny*nz
                	val2=-0.5*d0[6]
			#====== write binary file
			data=struct.pack("2id",row,col2,val2)
                        #data=struct.pack("2if",row,col2,val2)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col2))+' '+str(val2)+'\n')
                            #fout2.write(str(int(col2))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col2
                            damp_value[nnz_counter]=val2
                            nnz_counter=nnz_counter+1

                	col3=idmap(ix,iy,iz+2)
                	col3=col3+ip*nx*ny*nz
                	val3=-0.5*d0[6]
			#====== write binary file
			data=struct.pack("2id",row,col3,val3)
                        #data=struct.pack("2if",row,col3,val3)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col3))+' '+str(val3)+'\n')
                            #fout2.write(str(int(col3))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col3
                            damp_value[nnz_counter]=val3
                            nnz_counter=nnz_counter+1

                	col4=idmap(ix,iy+2,iz+2)
                	col4=col4+ip*nx*ny*nz
                	val4=0.5*d0[6]
			#====== write binary file
			data=struct.pack("2id",row,col4,val4)
                        #data=struct.pack("2if",row,col4,val4)
			row_data_bin.write(data)
			#===== store value
			if(ip==0):
                            #fout1.write(str(int(row))+' '+str(int(col4))+' '+str(val4)+'\n')
                            #fout2.write(str(int(col4))+'\n')
                            row_idx[nnz_counter]=row
                            col_idx[nnz_counter]=col4
                            damp_value[nnz_counter]=val4
                            nnz_counter=nnz_counter+1

                	#fout1.write('4 '+str(col1)+' '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+' '+str(col4)+' '+str(val4)+'\n')
			#fout1.write(str(col1)+' 7 '+str(val1)+' '+str(col2)+' '+str(val2)+' '+str(col3)+' '+str(val3)+' '+str(col4)+' '+str(val4)+'\n')
                	row=row+1

#fout1.close()
#fout2.close()
#fout3.close()
row_info_bin.close()
row_data_bin.close()
print('Finish row format!!\n')

print('Strat col format!!\n')

#======= sort index =====
col_sort_idx=sorted(range(nnz_number), key=lambda k: col_idx[k])
print('Finish col sorting!!\n')
print('There are '+str(nnz_counter)+' non-zero number. Estimeted #'+str(nnz_number)+'\n')

#=======================
#fout1=open(sys.argv[3]+'.col','w')
#fout2=open(sys.argv[3]+'.col_info','w')
col_info_bin=open(sys.argv[3]+'_col_info.bin','wb')
col_data_bin=open(sys.argv[3]+'_col_data.bin','wb')

row_add=int(int(total_row/2))
col_add=int(nx*ny*nz)

#col0=int(0)  # begin col number
col_count=int(0)
col_num=col0  #current col number
print('Cols start from col '+str(col_num)+'!\n')
for icol in range(nnz_counter):
	#=========== check ==========
	if((int(icol)%2000000)==0):
		print("working on: "+str(col_num)+'/'+str(col_add*2)+' = '+str(float(col_num)/(col_add*2)*100.0)+'%\n')
	#===ascii output
	#fout1.write(str(row_idx[col_sort_idx[icol]])+' '+str(col_idx[col_sort_idx[icol]])+' '+str(damp_value[col_sort_idx[icol]])+'\n')
	#====== write binary file
	#print('current col:'+str(icol)+'\n')
	#data=struct.pack("2id",row_idx[col_sort_idx[icol]],col_idx[col_sort_idx[icol]],damp_value[col_sort_idx[icol]])
        #data=struct.pack("2if",row_idx[col_sort_idx[icol]],col_idx[col_sort_idx[icol]],damp_value[col_sort_idx[icol]])
        data=struct.pack("2id",row_idx[col_sort_idx[icol]],col_idx[col_sort_idx[icol]],float(damp_value[col_sort_idx[icol]]))
	col_data_bin.write(data)

	#========= count non-zeros in col
	if(col_num==col_idx[col_sort_idx[icol]]):
		col_count=col_count+1
	else:
		#===ascii output
		#fout2.write(str(col_num)+' '+str(col_count)+'\n')
		#====== write binary file
		info=struct.pack("i",col_count)
		col_info_bin.write(info)
		#===== re-set counter
		col_num=col_num+1
		col_count=int(1) #already 1 in the col

#===== print out last col info ====
#===ascii output
#fout2.write(str(col_num)+' '+str(col_count)+'\n')
#====== write binary file
info=struct.pack("i",col_count)
col_info_bin.write(info)


#****************** for np 2 ****************
print('start 2nd run!!  '+str(col_num)+'!\n')
col_count=int(0)
col_num=col0  #current col number
for icol in range(nnz_counter):
	#=========== check ==========
	if((int(icol)%2000000)==0):
		print("working on: "+str(col_num+col_add)+'/'+str(col_add*2)+' = '+str(float(col_num+col_add)/(col_add*2)*100.0)+'%\n')
	#===ascii output
	#fout1.write(str(row_idx[col_sort_idx[icol]]+row_add)+' '+str(col_idx[col_sort_idx[icol]]+col_add)+' '+str(damp_value[col_sort_idx[icol]])+'\n')
	#====== write binary file
	#data=struct.pack("2id",row_idx[col_sort_idx[icol]]+row_add,col_idx[col_sort_idx[icol]]+col_add,damp_value[col_sort_idx[icol]])
        #data=struct.pack("2if",row_idx[col_sort_idx[icol]]+row_add,col_idx[col_sort_idx[icol]]+col_add,damp_value[col_sort_idx[icol]])
        data=struct.pack("2id",row_idx[col_sort_idx[icol]]+row_add,col_idx[col_sort_idx[icol]]+col_add,float(damp_value[col_sort_idx[icol]]))
	col_data_bin.write(data)

	#========= count non-zeros in col
	if(col_num==col_idx[col_sort_idx[icol]]):
		col_count=col_count+1
	else:
		#===ascii output
		#fout2.write(str(col_num+col_add)+' '+str(col_count)+'\n')
		#====== write binary file
		info=struct.pack("i",col_count)
		col_info_bin.write(info)
		#===== re-set counter
		col_num=col_num+1
		col_count=int(1) #already 1 in the col

#===== print out last col info ====
#===ascii output
#fout2.write(str(col_num+col_add)+' '+str(col_count)+'\n')
#====== write binary file
info=struct.pack("i",col_count)
col_info_bin.write(info)

#fout1.close()
#fout2.close()
col_info_bin.close()
col_data_bin.close()

print('Finish col format!!\n')

