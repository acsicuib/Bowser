"""
Titles:
Deep Reinforcement Learning for Service Placement in Fog Computing
Learning where deploy services from a descentralized approach

"""
import os
import time
import json
import logging.config
import networkx as nx
from configparser import ConfigParser

from environment.manager import write_header

from yafs.core import Sim
from yafs.topology import Topology
from yafs.distribution import *
from yafs.placement import JSONPlacement


from environment.workload import DynamicWorkload
from environment.path_routing import DeviceSpeedAwareRouting
from environment.manager import BowserManager
from environment.yafsExtensions import create_applications_from_json

def generate_random_services_and_allocations(graph,ratioServicesDeployedOnNodes,debug=True):
    deploys = []
    if not debug:
        numberNodes = len(graph.nodes())
        numberServices = np.random.randint(1,numberNodes * ratioServicesDeployedOnNodes )

        available_space = nx.get_node_attributes(graph, "HwReqs")
        for i in range(numberServices):
            deploy = {}
            app = np.random.randint(1,len(apps_level)+1) #un minimo de 1
            deploy["app"] = app
            deploy["module_name"] = "%i_01"%app
            levels = apps_level[app].keys()
            candidate_found = False
            tries_node = 40
            #Get a candidate node to put an aaplication
            while tries_node > 0 and not candidate_found:
                candidate_node = np.random.randint(0, numberNodes)
                tries_level = 10
                # Get a random flavour how fit in the HW avalablespace
                while tries_level > 0 and not candidate_found:
                    level = np.random.choice(list(levels), 1)[0]
                    hw = apps_level[app][level][0]
                    if int(available_space[candidate_node])-hw >= 0:
                        available_space[candidate_node] = int(available_space[candidate_node])-hw
                        candidate_found = True
                    else:
                        tries_level -=1
                        if len(list(levels))==1:
                            tries_level = 0

                if not candidate_found:
                    tries_node -=1

            if candidate_found:
                deploy["id_resource"] = candidate_node
                deploy["level"] = level
                deploys.append(deploy)
            else:
                print("Impossible to find an initial deployment of services")
                return None
    else: #TODO debug part
        deploy = {}
        deploy["app"] = 1
        deploy["module_name"] = "%i_01" % 1
        deploy["id_resource"] = 2
        deploy["level"] = "large"
        deploys.append(deploy)
        deploy = {}
        deploy["app"] = 2
        deploy["module_name"] = "%i_01" % 2
        deploy["id_resource"] = 0
        deploy["level"] = "small"
        deploys.append(deploy)
        deploy = {}
        deploy["app"] = 3
        deploy["module_name"] = "%i_01" % 3
        deploy["id_resource"] = 0
        deploy["level"] = "medium"
        deploys.append(deploy)

    return {"initialAllocation":deploys}

def generate_random_users_and_their_placements(maxNumberOfUsers):
    numberOfUsers = np.random.randint(1, maxNumberOfUsers)
    users = []
    for i in range(maxNumberOfUsers):
        user = {}
        app = np.random.randint(1, len(apps) + 1)
        user["app"] = app
        user["message"] = "M.USER.APP.%i" % app
        user["start"] = 0
        user["lambda"] = 100
        user["id_resource"] = np.random.randint(0, len(t.G.nodes()))
        users.append(user)
    return users

if __name__ == '__main__':
    import logging.config
    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    datestamp = "X"

    fileName = "experiment_S.json"
    f = open(fileName,"r")
    experiments = json.load(f)
    f.close()

    for item in experiments:
        start_time = time.time()

        code = item["code"]
        name = item["caseName"]
        case_path = "cases/%s/" %code

        temporal_folder = "results/results_%s_%s" % (name, datestamp) + "/"
        temporal_folder_traces = temporal_folder + "traces/"
        print("Experiment: %s\nStoring simulation results at folder: %s"%(name,temporal_folder))

        try:
            os.makedirs(temporal_folder)
        except OSError:
            None
        try:
            os.makedirs(temporal_folder_traces)
        except OSError:
            None

        config = ConfigParser()
        config.read(case_path + 'config.ini')
        minNumberOfSamples = int(config.get('simulation', 'minNumberOfSamples'))
        seed = int(config.get('simulation', 'seed'))

        ratioServicesDeployedOnNodes = float(config.get('services', 'deployedOnNodes'))
        maxNumberOfUsers = int(config.get('users', 'maxNumberOfUsers'))

        random.seed(seed)
        np.random.seed(seed)

        # Load the infrastructure aka topology
        t = Topology()
        dataNetwork = json.load(open(case_path + 'configuration/topology.json'))
        t.load_all_node_attr(dataNetwork)
        nx.write_gexf(t.G, temporal_folder + "topology.gexf") #debug

        # Applications
        dataApp = json.load(open(case_path + 'configuration/appDefinition.json'))
        apps = create_applications_from_json(dataApp)
        apps_level = {app["name"]: app["level"] for app in dataApp}

        # Routing algorithm
        routingPath = DeviceSpeedAwareRouting()

        # Perform changes in the environmnet for each execution
        bizarre_situations = 0
        state = 0
        currentNumberOfSamples = 0

        file_samples = "samples.csv"
        write_header(temporal_folder + file_samples)

        while currentNumberOfSamples < minNumberOfSamples:
            print("Launching one simulation")
            routingPath.clear_routing_cache() # or we can init the variable again
            data_placements = generate_random_services_and_allocations(t.G,ratioServicesDeployedOnNodes,debug=False) #TODO a debug control

            if data_placements != None:
                placements = JSONPlacement(name="Placement", json=data_placements)
            else:
                if bizarre_situations == 1000:
                    import sys
                    print("Something goes wrong. It was impossible to find a initial allocation of services.")
                    sys.exit(-1)
                bizarre_situations +=1
                continue



            # Simulation engine
            path_csv_files = temporal_folder_traces + "traces_%s_%i" % (name, state)
            s = Sim(t, default_results_path=path_csv_files)
            s.set_apps_levels(apps_level)

            # Initial app deployment in the sim engine according with the placement
            for aName in apps.keys():
                s.deploy_app(apps[aName], placements, routingPath)

            # Initial user placement & User request distribution & User deployment for each app
            dataPopulation = generate_random_users_and_their_placements(maxNumberOfUsers)
            for aName in apps.keys():
                data = []
                for element in dataPopulation:
                    if element['app'] == aName:
                        data.append(element)
                # For each application we create a DES process who introduce the users in the system
                # Each DES follows a distribution, in this case all these apps starts at the same simulation time: 1
                distribution = deterministic_distribution(name="DET", time=1)
                pop_app = DynamicWorkload(name="Dynamic_%s" % aName, data=data, iteration=state, activation_dist=distribution)
                s.deploy_pop(apps[aName], pop_app)


            """
            BOWSER service controller
            """
            bowserPeriod = int(config.get('BowerManager', 'activation_period'))
            trigger_distribution = deterministic_distribution(name="DET",time=bowserPeriod)

            bowser = BowserManager(path_csv_traces = path_csv_files,
                                   path_results=temporal_folder,
                                   apps_level = apps_level,
                                   samples_file=file_samples,
                                   iteration=state
                                   )

            s.deploy_monitor("App-Operator", bowser, trigger_distribution,
                             **{"sim": s,
                                "routing": routingPath,
                                "case_folder": case_path + "configuration/",
                                }
                             )

            # the magic begins!
            s.run(10000000000) # Bowser stops the simulation when check all available services per operation
            #TODO fix the number in function of the services x operations x period

            state += 1
            currentNumberOfSamples += bowser.number_samples

            # if currentNumberOfSamples == 1: # TODO DEBUG
            #     print("FINNITIQUITOOPOOOOO")
            #     numberOfSamples = 1

        print("\n\t--- %s seconds ---" % (time.time() - start_time))

    #end for experiments
    print("The end!")

