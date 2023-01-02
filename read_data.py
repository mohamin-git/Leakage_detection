from tokenize import Double
import pandas
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


"""
This File is used to import all the data from the csv and EPANET file and transform it pandas dataframe, to be processed in a next Step
"""
class water_network:
    def __init__(self, pressures, demands, flows, levels, junctions, pipes, leakages, cordinates) :
        self.pressures = pressures
        self.demands = demands
        self.flows = flows
        self.levels = levels
        self.junctions = junctions
        self.pipes = pipes
        self.leakages = leakages
        self.cordinates = cordinates



def read_file(filename):
    pressures= pandas.read_csv(filename + "_Pressures.csv", sep=";", decimal=',')
    demands = pandas.read_csv(filename + "_Demands.csv", sep=";", decimal=',')
    flows = pandas.read_csv(filename + "_Flows.csv", sep=";", decimal=',')
    levels = pandas.read_csv(filename + "_Levels.csv", sep=";", decimal=',')
    leakages = pandas.read_csv(filename + "_Leakages.csv", sep=";", decimal= ",")
    return pressures, demands, flows, levels, leakages



def read_inp_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    junction_section= False
    junctions =  []
    pipes_section= False
    pipes =  []
    cordinates_section= False
    cordinates =  []
    patterns_section= False
    patterns = []
    for idx, line in enumerate(lines):
        if "[JUNCTIONS]" in line:
            #cols_names = lines[idx+1].replace(" ", "").replace("\n","").split("\t")
            #junctions.append(cols_names)
            junction_section = True
        elif junction_section and len(line) != 1:
            if len(junctions) == 0 :
                junctions.append(lines[idx].replace(" ", "").replace("\n","").split("\t"))
                junctions[0].append("sensors")
            else:    
                junctions.append(lines[idx].replace(" ", "").replace("\n","").split("\t"))
        elif len(line) <= 1 & junction_section:
            junction_section = False

        if "[PIPES]" in line:
            pipes_section = True
        elif pipes_section and len(line) != 1:
            if len(pipes) == 0 :
                pipes.append(lines[idx].replace(" ", "").replace("\n","").split("\t"))
                pipes[0].append("sensors") 
            else:    
                pipes.append(lines[idx].replace(" ", "").replace("\n","").split("\t"))
        elif len(line) <= 1 & pipes_section:
            pipes_section = False   
            
        if "[COORDINATES]" in line:
            cordinates_section = True
        elif cordinates_section and len(line) != 1 :
            cordinates.append(lines[idx].replace(" ", "").replace("\n","").split("\t"))
        elif len(line) <= 1 & cordinates_section:
            cordinates_section = False      
            
            
        if "[PATTERNS]" in line:
            patterns_section = True
        elif patterns_section and len(line)  > 2 and "P-Industrial" not in line:
            patterns.append(lines[idx].replace(" ", "").replace("\n","").split("\t"))
        elif len(line) <= 1 & patterns_section:
            patterns_section = False  
    #print(patterns)
    #for line in patterns:
    #   if len(line) !=   7:
    #        print(line)
    patterns[0] += ["Multipliers1","Multipliers2","Multipliers3","Multipliers4", "Multipliers5"]        
    junctions = np.array(junctions)
    patterns = np.array(patterns)
    pipes =np.array(pipes)
    cordinates = np.array(cordinates)
    cordinates = pandas.DataFrame(cordinates[1:], columns=cordinates[0])
    cordinates.set_index(';Node', inplace=True)
    patterns = pandas.DataFrame(patterns[1:], columns=patterns[0])
    junctions = pandas.DataFrame(junctions[1:], columns=junctions[0])
    junctions.set_index(';ID', inplace=True)
    pipes = pandas.DataFrame(pipes[1:], columns=pipes[0])
    pipes.set_index(';ID', inplace=True)
    return junctions, pipes, cordinates, patterns


