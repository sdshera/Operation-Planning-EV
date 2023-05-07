#Srijan_Dasgupta EV Charging pool
# -*- coding: utf-8 -*-

!pip install pyomo
!wget -N -q "https://ampl.com/dl/open/cbc/cbc-linux64.zip"
!unzip -o -q cbc-linux64


import pandas as pd

"""## EV Class"""

import random

class EV():
    def __init__(self): #add initial_EV_state as input??   

        #For now, no idea what these two attributes are

        self.Pa  = 0
        self.Pb = 3    
        self.eta1 = 0.8
        self.eta2 = 1.0
        self.eta3 = 0.7
        self.cost_weight = 64.0


        self.arrival_time = random.randrange(6, 10)  # h
        self.departure_time = self.arrival_time + random.randrange(7, 10)  # h
        self.arrival_soc = random.randrange(60, 95) / 100  # %
        self.departure_soc = random.randrange(90, 99) / 100  # %, max soc assumed 100 %  #desired departure state of charge
        
        self.state = 'What is the status of the car at the current simulation ? arrived / not arrived / not connected ' #initial_EV_state #initial_EV_state --> this hasn't been defined
        
        self.capacity = random.randrange(50, 70)
        self.max_charging_power = random.sample([3, 6, 11], 1)[0]
        self.price = random.randrange(5, 10)*self.capacity

"""## Input Data to the Controller"""

number_of_EVs = 3

SoC_init = {0: 0.0, 1: 0.0, 2: 0.83}

control_horizon = 20

current_time = 7

car0 = EV() #{EV}

car0.Pa = 0.1
car0.Pb = 0.7
car0.arrival_soc = 0.9
car0.arrival_time = 8
car0.capacity = 56
car0.cost_weight = 64.0
car0.departure_soc = 0.9
car0.departure_time = 14
car0.eta1 = 0.8
car0.eta2 = 1.0
car0.eta3 = 0.7
car0.max_charging_power = 6
car0.price = 448
car0.state = 'not_arrived'


car1 = EV()
car1.Pa = 0.1
car1.Pb = 0.7
car1.arrival_soc = 0.7
car1.arrival_time = 8
car1.capacity = 66
car1.cost_weight = 63.0
car1.departure_soc = 0.9
car1.departure_time = 17
car1.eta1 = 0.8
car1.eta2 = 1.0
car1.eta3 = 0.7
car1.max_charging_power = 6
car1.price = 462
car1.state = 'not_arrived'

car2 = EV() #{EV} 
car2.Pa = 0.1
car2.Pb = 0.7
car2.arrival_soc = 0.83
car2.arrival_time = 6
car2.capacity = 63
car2.cost_weight = 63.0
car2.departure_soc = 0.9
car2.departure_time = 13
car2.eta1 = 0.8
car2.eta2 = 1.0
car2.eta3 =  0.7
car2.max_charging_power = 3
car2.price = 567
car2.state = 'connected' 

ev = [car0, car1, car2] #{list: 3}


exp_price = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]

forecast_PV = [0.2985, 0.5502333333333334, 1.0197, 2.0184333333333333, 2.9785, 2.093333333333333, 3.3649, 3.9546333333333332, 4.3033, 1.0353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

grid_capacity = 10

imp_price = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

for e in range (number_of_EVs):
  print(ev[e].price)

"""# Controller

**You have to submit the code of the cell below on Gradescope as a single file named** `controller.py`
"""

# Version of 2021-12-14 23:00

from pyomo.environ import *

class ChargeController:

    def compute_actions(self, number_of_EVs, control_horizon, grid_capacity, current_time, forecast_PV, imp_price, exp_price, ev, prev_soc): #prev_soc
        """
        Function that computes the controller's actions regarding the charging power of the EVs, expressed as a percentage of the maximum charging power.

        :param number_of_EVS: look-ahead control horizon (hour)
        :param control_horizon: look-ahead control horizon (hour)
        :param grid_capacity: Power exchange limitation with the grid (kW)
        :param current_time: period considered, expressed as the hour of the day at which the optimal operation of the problem should start (hour)
        :param forecast_PV: Forecast of photovoltaic production over the control horizon (list[(kW)])
        :param imp_price: Import price over the control horizon (list[(€/kWh)]
        :param exp_price: Export price over the control horizon (list[(€/kWh)]
        :param ev: List of EV objects (see class EV)
        :param prev_soc: Dict of state of charge of the EVs at the previous time step (Dict[ev] = %)
        :return: EV charging setpoint (% maximum charging power) and PV generation over the control_horizon (What is this PV generation?)
        """
        
        eff = 0.9275

        imp_price = { i : imp_price[i] for i in range(0, len(imp_price) ) }
        impr_price_dict = dict()
        for i in range(control_horizon):
          impr_price_dict[i] = imp_price[i]


        forecast_PV = { i : forecast_PV[i] for i in range(0, len(forecast_PV) ) }
        forecast_PV_dict = dict()
        for i in range(control_horizon):
          forecast_PV_dict[i] = forecast_PV[i]


        exp_price = { i : exp_price[i] for i in range(0, len(exp_price) ) }
        exp_price_dict = dict()
        for i in range(control_horizon):
          exp_price_dict[i] = exp_price[i]


        # Initializing return values  

        Dt = 1 #time step of 1h

        # Note: the simulator is expecting "control_horizon" values, but will use only the first one, so you can pad with zeros.
        
        #output 1
        charge_powers = dict()
       
        for e in range(number_of_EVs):
          charge_powers[e] = [0.0] * control_horizon

        #output 2
        PV_generation = [0.0] * control_horizon      #real power usage from PV 
         
              
        
        model = ConcreteModel()
        model.IDX = Set(initialize = range(control_horizon)) #index iterable over the control_horizon i--> 7, 14
        model.cDX = Set(initialize = range(number_of_EVs))   #0,3

        model.imp_price = Param(model.IDX, initialize = impr_price_dict)
        model.exp_price = Param(model.IDX, initialize = exp_price_dict)
        model.cost_PV_generation = Param(model.IDX, initialize = 0.0)
        model.trade_off_coefficient = Param(model.IDX, initialize = 100)  #should this coefficient be bound between 0 and 1?
        model.ev = Param(model.cDX, initialize = ev)
        model.forecast_PV = Param(model.IDX, initialize = forecast_PV_dict)

        model.charge_powers = Var( model.cDX, model.IDX, domain=NonNegativeReals)
        model.PV_used = Var(model.IDX, domain = NonNegativeReals)
        model.PV_exported = Var(model.IDX, domain = NonNegativeReals)
        model.state_of_charge = Var(model.cDX, model.IDX, domain = NonNegativeReals) #bounds or not
        model.imp_quantity = Var(model.IDX, domain = NonNegativeReals)
        model.exp_quantity = Var(model.IDX, domain = NonNegativeReals)
        model.bin_export_import = Var(model.IDX, domain=Binary)
        model.pos = Var(model.cDX, domain = NonNegativeReals)
        model.neg = Var(model.cDX, domain = NonNegativeReals)


        def obj_expression(model):

          return sum((model.imp_price[i] * model.imp_quantity[i] - model.exp_price[i] * model.exp_quantity[i]) * Dt for i in model.IDX) 
          + model.trade_off_coefficient \
          * sum( (model.pos[car] + model.neg[car]) * (model.ev[car].departure_time - model.ev[car].arrival_time)\
           * model.ev[car].price / model.ev[car].capacity for car in model.cDX)
          
                    
        model.obj = Objective(rule = obj_expression, sense = minimize)


        def equilibrium_constraint(model, i):
          return model.imp_quantity[i] + model.PV_used[i]   == model.exp_quantity[i] + sum(charge_powers[car][i] for car in model.cDX)

        def PV_power_constraint(model, i):
          return model.exp_quantity[i] == model.forecast_PV[i] - model.PV_used[i]


        def avoid_import_export1(model, i):
            return model.imp_quantity[i] <= grid_capacity * model.bin_export_import[i]

        def avoid_import_export2(model, i):
            return model.exp_quantity[i] <= grid_capacity * (1 - model.bin_export_import[i])



        #def power_transfer_constraint(model, i):
            #return  value(model.imp_quantity[i]) <= grid_capacity #and value(model.exp_quantity[i]) <= grid_capacity )

        #def max_power_allocated(model, car, i):
          #return value(charge_powers[car][i]) <=  1

        #def max_desired_soc(model, i):                 set constraint on departure soc or not
          #return (<= ev[e].departure_soc for 

        #def bound_on_soc(model, car, i):
          #return model.ev[car].arrival_soc  +  sum(charge_powers[car][i]*model.ev[car].max_charging_power * Dt / model.ev[car].capacity) <= 1
      
        def state_of_charge_constraint(model, i, car):
          if i == 0:
            if model.ev[car].state != "connected":
              return model.state_of_charge[car, i] == model.ev[car].arrival_soc
            else:
              return model.state_of_charge[car, i] == model.ev[car].arrival_soc + model.charge_powers[car, i] * eff * Dt / model.ev[car].capacity

              
          else:
            if i < model.ev[car].arrival_time - current_time:
              return model.state_of_charge[car, i] == model.state_of_charge[car, i-1]

            elif i == model.ev[car].arrival_time - current_time:
              return model.state_of_charge[car, i] == model.ev[car].arrival_soc + model.charge_powers[car, i] * eff * Dt / model.ev[car].capacity

            elif i > model.ev[car].arrival_time - current_time and i <= model.ev[car].departure_time - current_time:
              return model.state_of_charge[car, i] == model.state_of_charge[car, i-1] + model.charge_powers[car, i] * eff * Dt / model.ev[car].capacity

            else:
              return model.state_of_charge[car, i] == model.state_of_charge[car, model.ev[car].departure_time - current_time ]


         #return model.state_of_charge[e][i] == model.state_of_charge[e][i -1] + (charge_powers[e][i]*ev[e].max_charging_power / ev[e][i].capacity) * (ev[e].departure_time - ev[e].arrival_time)


        #def energy_constraint(model, car, i):
          #return model.state_of_charge[car][i] * model.ev[car].capacity  +  charge_powers[car][i]*model.ev[car].max_charging_power*Dt <= model.ev[car].capacity

        def absolute_value_constraint(model, car):
          if model.ev[car].departure_time - current_time >= 0:
            return model.pos[car] - model.neg[car]  == model.state_of_charge[car, model.ev[car].departure_time - current_time ] - model.ev[car].departure_soc \
                      
          else:
            return model.pos[car] + model.neg[car] == 0


        
        model.constraint1 = Constraint(model.IDX, rule=equilibrium_constraint)
        model.constraint2 = Constraint(model.IDX, rule = PV_power_constraint)
        model.constraint3 = Constraint(model.IDX, rule = avoid_import_export1)
        model.constraint4 = Constraint(model.IDX, rule = avoid_import_export2)
        model.constraint5 = Constraint(model.IDX, model.cDX, rule = state_of_charge_constraint)
        model.constraint6 = Constraint(model.cDX, rule = absolute_value_constraint)

        model.constraint7 = ConstraintList()   #max power allocated constraint
        for car in model.cDX:
          for i in model.IDX:
            model.constraint7.add(expr = model.charge_powers[car,i] <= model.ev[car].max_charging_power) #model.ev[car].max_charging_power





        # Code snipet for calling the solver 
        # Move this code where needed.
        solver = SolverFactory("cbc")
        results = solver.solve(model, tee=False) # tee=True makes the solver verbose

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            pass # Do something when the solution is optimal and feasible
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print (">>> INFEASIBLE MODEL dumped to tmp.lp")
            model.write("tmp.lp", io_options={'symbolic_solver_labels': True}) # Export the model
            # TODO Should try a fallback strategy to avoid crashing
        else:
            # Something else is wrong
            print("Solver Status: ",  results.solver.status)
            print (">>> MODEL dumped to strange.lp")
            model.write("strange.lp", io_options={'symbolic_solver_labels': True}) # Export the model

        for i in model.IDX:
          for car in model.cDX:
            charge_powers[car, i] = (value(model.charge_powers[car, i]) / value(model.ev[car].max_charging_power) )
            PV_generation[i] = value(model.PV_used[i])
            


        #Printing data to the console


        SOC = [[value(model.state_of_charge[car,i]) for i in model.IDX] for car in model.cDX]

        imprt = []
        export = []

        for i in model.IDX:
            PV_generation[i] = value(model.PV_used[i])
            imprt.append(value(model.imp_quantity[i]))
            export.append( value(model.exp_quantity[i]))
            
            
        print('SOC ev0')
        for i in model.IDX:
            print(value(model.state_of_charge[0,i]))
        
        print('--------------------------------------------------------------------------------')

        print('SOC ev1')
        for i in model.IDX:
            print(value(model.state_of_charge[1,i]))

        print('--------------------------------------------------------------------------------')

        print('SOC ev2')
        for i in model.IDX:
            print(value(model.state_of_charge[2,i]))

        print('--------------------------------------------------------------------------------')

        car0_charge_power = list()
        for i in model.IDX:
            car0_charge_power.append(value(model.charge_powers[0,i]))
            
        car1_charge_power = list()
        for i in model.IDX:
            car1_charge_power.append(value(model.charge_powers[1,i]))
            
        car2_charge_power = list()
        for i in model.IDX:
            car2_charge_power.append(value(model.charge_powers[2,i]))
            
        power_in_cars = list()
        for i in model.IDX:
            power_in_cars.append(sum(value(model.charge_powers[car,i]) for car in model.cDX))
            

        hours = list()
        for i in range(current_time, current_time + control_horizon):
            hours.append(i)
        data = {'Hour': hours,'Import': imprt, 'Export':export, 'PV generation':PV_generation, 'PV Forecast': list(forecast_PV_dict.values()) , \
                  'EV_0 Pow': car0_charge_power, 'EV_1 Pow': car1_charge_power,'EV_2 Pow': car2_charge_power, }

        data2 = {'Hour': hours,'Import': imprt, 'Export':export, 'PV generation':PV_generation, 'PV availability': list(forecast_PV_dict.values()), "Power in Cars": power_in_cars}    
            
        df = pd.DataFrame(data)
        df2 = pd.DataFrame(data2)

        print (df)
        print(df2)

        




       

        return charge_powers, PV_generation # Do not change the order

# Version of 2021-12-14 23:00

from pyomo.environ import *

class ChargeController:

    def compute_actions(self, number_of_EVs, control_horizon, grid_capacity, current_time, forecast_PV, imp_price, exp_price, ev, SoC_init): #prev_soc
        """
        Function that computes the controller's actions regarding the charging power of the EVs, expressed as a percentage of the maximum charging power.

        :param number_of_EVS: look-ahead control horizon (hour)
        :param control_horizon: look-ahead control horizon (hour)
        :param grid_capacity: Power exchange limitation with the grid (kW)
        :param current_time: period considered, expressed as the hour of the day at which the optimal operation of the problem should start (hour)
        :param forecast_PV: Forecast of photovoltaic production over the control horizon (list[(kW)])
        :param imp_price: Import price over the control horizon (list[(€/kWh)]
        :param exp_price: Export price over the control horizon (list[(€/kWh)]
        :param ev: List of EV objects (see class EV)
        :param prev_soc: Dict of state of charge of the EVs at the previous time step (Dict[ev] = %)
        :return: EV charging setpoint (% maximum charging power) and PV generation over the control_horizon (What is this PV generation?)
        """
        
        eff = 0.9275 #this efficiency was calculated using the fucntion of the efficiencies that was given to us in the figure
        #the import price, export price and the forecast pv is going to be given by the website, thus we need to create a dictionary of these values using the same length of the 
        imp_price = { i : imp_price[i] for i in range(0, len(imp_price) ) } 
        impr_price_dict = dict()
        for i in range(control_horizon):
          impr_price_dict[i] = imp_price[i]


        forecast_PV = { i : forecast_PV[i] for i in range(0, len(forecast_PV) ) }
        forecast_PV_dict = dict()
        for i in range(control_horizon):
          forecast_PV_dict[i] = forecast_PV[i]


        exp_price = { i : exp_price[i] for i in range(0, len(exp_price) ) }
        exp_price_dict = dict()
        for i in range(control_horizon):
          exp_price_dict[i] = exp_price[i]


        # Initializing return values  

        Dt = 1 #time step of 1h

        # Note: the simulator is expecting "control_horizon" values, but will use only the first one, so you can pad with zeros.
        
        #output 1
        charge_powers = dict()
       
        for e in range(number_of_EVs):
          charge_powers[e] = [0.0] * control_horizon

        #output 2
        PV_generation = [0.0] * control_horizon      #real power usage from PV 
         
              
        
        model = ConcreteModel()
        model.IDX = Set(initialize = range(control_horizon)) #index iterable over the control_horizon i--> 7, 14
        model.cDX = Set(initialize = range(number_of_EVs))   #0,3

        model.imp_price = Param(model.IDX, initialize = impr_price_dict)
        model.exp_price = Param(model.IDX, initialize = exp_price_dict)
        model.cost_PV_generation = Param(model.IDX, initialize = 0.0)
        model.trade_off_coefficient = Param(model.IDX, initialize = 100)  #should this coefficient be bound between 0 and 1?
        model.ev = Param(model.cDX, initialize = ev)
        model.forecast_PV = Param(model.IDX, initialize = forecast_PV_dict)

        model.charge_powers = Var( model.cDX, model.IDX, domain=NonNegativeReals)
        model.PV_used = Var(model.IDX, domain = NonNegativeReals)
        model.PV_exported = Var(model.IDX, domain = NonNegativeReals)
        model.state_of_charge = Var(model.cDX, model.IDX, domain = NonNegativeReals) #bounds or not
        model.imp_quantity = Var(model.IDX, domain = NonNegativeReals)
        model.exp_quantity = Var(model.IDX, domain = NonNegativeReals)
        model.bin_export_import = Var(model.IDX, domain=Binary)
        model.pos = Var(model.cDX, domain = NonNegativeReals)
        model.neg = Var(model.cDX, domain = NonNegativeReals)


        def obj_expression(model):

          return sum((model.imp_price[i] * model.imp_quantity[i] - model.exp_price[i] * model.exp_quantity[i]) * Dt for i in model.IDX) 
          + model.trade_off_coefficient \
          * sum( (model.pos[car] + model.neg[car]) * (model.ev[car].departure_time - model.ev[car].arrival_time)\
           * model.ev[car].price / model.ev[car].capacity for car in model.cDX)
          
                    
        model.obj = Objective(rule = obj_expression, sense = minimize)


        def equilibrium_constraint(model, i):
          return model.imp_quantity[i] + model.PV_used[i]   == model.exp_quantity[i] + sum(charge_powers[car][i] for car in model.cDX)

        def PV_power_constraint(model, i):
          return model.exp_quantity[i] == model.forecast_PV[i] - model.PV_used[i]


        def avoid_import_export1(model, i):
            return model.imp_quantity[i] <= grid_capacity * model.bin_export_import[i]

        def avoid_import_export2(model, i):
            return model.exp_quantity[i] <= grid_capacity * (1 - model.bin_export_import[i])



        #def power_transfer_constraint(model, i):
            #return  value(model.imp_quantity[i]) <= grid_capacity #and value(model.exp_quantity[i]) <= grid_capacity )

        #def max_power_allocated(model, car, i):
          #return value(charge_powers[car][i]) <=  1

        #def max_desired_soc(model, i):                 set constraint on departure soc or not
          #return (<= ev[e].departure_soc for 

        #def bound_on_soc(model, car, i):
          #return model.ev[car].arrival_soc  +  sum(charge_powers[car][i]*model.ev[car].max_charging_power * Dt / model.ev[car].capacity) <= 1
      
        def state_of_charge_constraint(model, i, car):
          if i == 0:
            if model.ev[car].state != "connected":
              return model.state_of_charge[car, i] == model.ev[car].arrival_soc
            else:
              return model.state_of_charge[car, i] == model.ev[car].arrival_soc + model.charge_powers[car, i] * eff * Dt / model.ev[car].capacity

              
          else:
            if i < model.ev[car].arrival_time - current_time:
              return model.state_of_charge[car, i] == model.state_of_charge[car, i-1]

            elif i == model.ev[car].arrival_time - current_time:
              return model.state_of_charge[car, i] == model.ev[car].arrival_soc + model.charge_powers[car, i] * eff * Dt / model.ev[car].capacity

            elif i > model.ev[car].arrival_time - current_time and i <= model.ev[car].departure_time - current_time:
              return model.state_of_charge[car, i] == model.state_of_charge[car, i-1] + model.charge_powers[car, i] * eff * Dt / model.ev[car].capacity

            else:
              return model.state_of_charge[car, i] == model.state_of_charge[car, model.ev[car].departure_time - current_time ]


         #return model.state_of_charge[e][i] == model.state_of_charge[e][i -1] + (charge_powers[e][i]*ev[e].max_charging_power / ev[e][i].capacity) * (ev[e].departure_time - ev[e].arrival_time)


        def energy_constraint(model, car, i):
          return model.state_of_charge[car][i] * model.ev[car].capacity  +  charge_powers[car][i]*model.ev[car].max_charging_power*Dt <= model.ev[car].capacity

        def absolute_value_constraint(model, car):
          if model.ev[car].departure_time - current_time >= 0:
            return model.pos[car] - model.neg[car]  == model.state_of_charge[car, model.ev[car].departure_time - current_time ] - model.ev[car].departure_soc \
                      
          else:
            return model.pos[car] + model.neg[car] == 0


        
        model.constraint1 = Constraint(model.IDX, rule=equilibrium_constraint)
        model.constraint2 = Constraint(model.IDX, rule = PV_power_constraint)
        model.constraint3 = Constraint(model.IDX, rule = avoid_import_export1)
        model.constraint4 = Constraint(model.IDX, rule = avoid_import_export2)
        model.constraint5 = Constraint(model.IDX, model.cDX, rule = state_of_charge_constraint)
        model.constraint6 = Constraint(model.cDX, rule = absolute_value_constraint)

        model.constraint7 = ConstraintList()   #max power allocated constraint
        for car in model.cDX:
          for i in model.IDX:
            model.constraint7.add(expr = model.charge_powers[car,i] <= model.ev[car].max_charging_power) #model.ev[car].max_charging_power





        # Code snipet for calling the solver 
        # Move this code where needed.
        solver = SolverFactory("cbc.exe")
        results = solver.solve(model, tee=False) # tee=True makes the solver verbose

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            pass # Do something when the solution is optimal and feasible
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print (">>> INFEASIBLE MODEL dumped to tmp.lp")
            model.write("tmp.lp", io_options={'symbolic_solver_labels': True}) # Export the model
            # TODO Should try a fallback strategy to avoid crashing
        else:
            # Something else is wrong
            print("Solver Status: ",  results.solver.status)
            print (">>> MODEL dumped to strange.lp")
            model.write("strange.lp", io_options={'symbolic_solver_labels': True}) # Export the model

        for i in model.IDX:
          for car in model.cDX:
            charge_powers[car, i] = (value(model.charge_powers[car, i]) / value(model.ev[car].max_charging_power) )
            PV_generation[i] = value(model.PV_used[i])
            


        #Printing data to the console


        SOC = [[value(model.state_of_charge[car,i]) for i in model.IDX] for car in model.cDX]

        imprt = []
        export = []

        for i in model.IDX:
            PV_generation[i] = value(model.PV_used[i])
            imprt.append(value(model.imp_quantity[i]))
            export.append( value(model.exp_quantity[i]))
            
            
        print('SOC ev0')
        for i in model.IDX:
            print(value(model.state_of_charge[0,i]))
        
        print('--------------------------------------------------------------------------------')

        print('SOC ev1')
        for i in model.IDX:
            print(value(model.state_of_charge[1,i]))

        print('--------------------------------------------------------------------------------')

        print('SOC ev2')
        for i in model.IDX:
            print(value(model.state_of_charge[2,i]))

        print('--------------------------------------------------------------------------------')

        car0_charge_power = list()
        for i in model.IDX:
            car0_charge_power.append(value(model.charge_powers[0,i]))
            
        car1_charge_power = list()
        for i in model.IDX:
            car1_charge_power.append(value(model.charge_powers[1,i]))
            
        car2_charge_power = list()
        for i in model.IDX:
            car2_charge_power.append(value(model.charge_powers[2,i]))
            
        power_in_cars = list()
        for i in model.IDX:
            power_in_cars.append(sum(value(model.charge_powers[car,i]) for car in model.cDX))
            

        hours = list()
        for i in range(current_time, current_time + control_horizon):
            hours.append(i)
        data = {'Hour': hours,'Import': imprt, 'Export':export, 'PV generation':PV_generation, 'PV Forecast': list(forecast_PV_dict.values()) , \
                  'EV_0 Pow': car0_charge_power, 'EV_1 Pow': car1_charge_power,'EV_2 Pow': car2_charge_power, "Sum of Power in Cars": power_in_cars}

        #data2 = {'Hour': hours,'Import': imprt, 'Export':export, 'PV generation':PV_generation, 'PV availability': list(forecast_PV_dict.values()), "Sum of Power in Cars": power_in_cars}    
            
        df = pd.DataFrame(data)
        #df2 = pd.DataFrame(data2)

        print (df)
        #print(df2)
        
        print('--------------------------------------------------------------------------------')
        




       

        return charge_powers, PV_generation # Do not change the order

"""## Calling the Controller"""

myController = ChargeController()
myController.compute_actions(number_of_EVs, control_horizon, grid_capacity, current_time, forecast_PV, imp_price, exp_price, ev, SoC_init)

"""# EV Object"""

import random

class EV():
    def __init__(self): #add initial_EV_state as input??   

        self.arrival_time = random.randrange(6, 10)  # h
        self.departure_time = self.arrival_time + random.randrange(7, 10)  # h
        self.arrival_soc = random.randrange(60, 95) / 100  # %
        self.departure_soc = random.randrange(90, 99) / 100  # %, max soc assumed 100 %  #desired departure state of charge
        
        self.state = 'initial_EV_state' #initial_EV_state --> this hasn't been defined
        
        self.capacity = random.randrange(50, 70)
        self.max_charging_power = random.sample([3, 6, 11], 1)[0]
        self.price = random.randrange(5, 10)*self.capacity




for car in range(5):
  tesla = EV()
  #print('Price: ' + str(tesla.price), 'Arrival Time: ' + str(tesla.arrival_time))
  print(tesla.max_charging_power)

