import math
import os
import random
import pandas as pd
import sys
import time
import traceback
from subprocess import Popen



# import os
# info = open("info.csv", 'w')
# directory = r'graphs/exact_public/'
# for filename in os.listdir(directory):
#     f = open(directory+filename, 'r')
#     variable = f.readline()
#     variable = variable.replace(' ',',')
#     info.write(filename+','+variable)
#     f.close()
#
# info.close()
# files =['e_001','e_059','e_033','e_189','e_039','e_025','e_091','e_083','e_149','e_187','e_017','e_191','e_007','e_067','e_061','e_071','e_023','e_183','e_121','e_167' ]

# try:
#     import time
#     time.sleep(1)
#     print(os.getenv('TEST_VARIABLE'),' ',sys.argv[1])
#     raise Exception("hey")
#     time.sleep(10)
# except Exception as e:
#     with open('error.log', 'a+') as f:
#         f.write('\n')
#         f.write("\n")
#         f.write(traceback.format_exc())
#         f.write(str(e))
#     f.close()


#
# def load_graph_obj_from_file(path_to_graph):
#     # graph is structured as follows:
#     """
#     1. <number of vertices> <number of edges> 0
#     2. <vertex num> <vertex num> <vertex num> ... #this is all edges from 1 (2-1) to <vertex num>
#     3. " "                                         # same as above but from two
#     ..
#     ..
#     <number of vertices> + 1. <vertex num> <vertex num> #this is all edges from vertex <number of vertices> to <vertex num>
#     # blank line at the end
#     """
#     with open(path_to_graph, 'r+') as file_in:
#         file = file_in.read()
#     file = file.split('\n')
#     line1 = file[0].split(' ')
#     print (path_to_graph)
#     number_of_vertices = line1[0]
#     number_of_edges = line1[1]
#     with open('info.csv', 'a+') as f:
#         f.write('\n')
#         f.write(path_to_graph+ ','+number_of_vertices+','+number_of_edges)
#
#     f.close()
#     file_in.close()
#     return
#
# dir1 ='graphs/exact_public/'
# dir2 ='graphs/exact_public_2/'
# for file in os.listdir(dir1):
#     load_graph_obj_from_file(dir1+file)
#
# for file in os.listdir(dir2):
#     load_graph_obj_from_file(dir2+file)

# import pandas as pd
# df = pd.read_csv('info.csv')
# df['ratio'] = df['E']/df['V']
# df['ratio*V'] = df['ratio']*df['E']
#
from discord_webhook import DiscordWebhook


def send_discord_message(message):
    url = os.getenv('discord_webhook_url')
    webhook = DiscordWebhook(url=url, rate_limit_retry=True, content=message)
    return webhook.execute()

if __name__ == '__main__':
    df = pd.read_csv(os.getcwd()+'/sccs_emperical.csv')
    go = True
    good = '4676_e_093'
    for index, rows in df.iterrows():
        path = os.getcwd()+'/graphs/sccs/'+rows['graph_name']

        if go:
            try:
                os.system('python3 ExactSolver.py '+ path)
            except Exception as e:
                send_discord_message('FAILURE')

        if rows['graph_name']==good:
            go = True

