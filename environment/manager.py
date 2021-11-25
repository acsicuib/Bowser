import logging
import pandas as pd
from collections import defaultdict
from yafs.topology import *
from yafs.distribution import *

# Issues:
# stay operation (N,S,A,N,F)

# OPERATIONS = ["undeploy", "nop", "migrate", "replicate", "shrink", "evict", "reject", "adapt"]
FLAVOURS = ["small", "medium", "large"]


# NOTE: flavours/adapt operations should be at the end of this list
OPERATIONS = ["small", "medium", "large"]
OPERATIONS = ["undeploy", "replicate", "migrate", "small", "medium","large"]

# OPERATIONSIONS = ["undeploy", "small", "medium", "large"]
# OPERATIONS = ["undeploy","replicate"]

# we can do more cases if
# OPERATIONS = [undeploy,migrate,migrate,...,migrate, replicate, replicate, ... , replicate]+FLAVOURS


# Samples measures order: cts, requests,node, Future_requests, Fnode,

# MEASURES = ["sim", #simulation id.
#             "app", "currentFlavour","action", # CTS
#             "HWreq",
#             "OdiffReqChannels", 'OreqChannels', 'OsumReq', 'OavgReq', 'OsumLat', 'OavgLat',  # Requests - Old State
#             "HWtotal", "HWfree", "utilization", "degree", "centrality", "nusers",  # Status of the current node
#             #FUTURE
#             "FHWreq",
#             "FdiffReqChannels", 'FreqChannels', 'FsumReq', 'FavgReq', 'FsumLat', 'FavgLat',  # Requests - Future State
#             "FHWtotal", "FHWfree", "Futilization", "Fdegree", "Fcentrality", "Fnusers",  # Status of the current node
#             "fit"
#             ]

MEASURES = [ "Sim","Action",
             #==== Service information in the current node previous to the action
            "App",
            "CurrentFlavour",
            "SupportedRequests",
            "HWReq",
            # ==== Request data
            "DifReqChannels",
            "TotReqChannels",
            "SumRequests",
            "AvgRequests",
            "SumLat",
            "AvgLat",
            # ==== Node data
            "HWTotal",
            "HWUsed",  #(included the current service)
            "Utilization",
            "Degree",
            "Centrality",
            "ConnectedUsers",#[0,n]
             #==== Destinate Node

            "DstHWUsed", #(not include the current service)
            "DstUtilization" ,#(not include the current service)

            # ----
            # Future State
            # ----
            # Service information in the new position
            #"NextFlavour",
            "NextSupportedRequests",
            "NextHWReq",
            "NextDifReqChannels",
            "NextTotReqChannels",
            "NextSumRequests",
            "NextAvgRequests",
            "NextSumLat",
            "NextAvgLat",

            # Impact of the new service in the current node
            "DstHWTotal",
            "NextHWUsed",# (including the current service)
            "NextUtilization",
            "DstDegree",
            "DstCentrality",
            "DstConnectedUsers",

            "Reward"
]

def write_header(file):
    with open(file, "w") as f:
        columns = ",".join([item for item in MEASURES])
        f.write(columns + "\n")

def deploy_module(sim, service, node_dst=None, replicate=False):
    app = sim.apps[service["app"]]
    servs = app.services
    if node_dst == None:
        service["des"] = sim.deploy_module(app_name=service["app"], module=service["service_name"],
                                           services=servs[service["service_name"]], ids=[service["node"]],
                                           level=service["level"])[0]
    else:
        if replicate:
            service["clone_des"] = sim.deploy_module(app_name=service["app"], module=service["service_name"],
                                                     services=servs[service["service_name"]], ids=[node_dst],
                                                     level=service["level"])[0]
        else:
            service["des"] = sim.deploy_module(app_name=service["app"], module=service["service_name"],
                                               services=servs[service["service_name"]], ids=[node_dst],
                                               level=service["level"])[0]


def undeploy_module(sim, service, replicate=False):
    try:
        if not replicate:
            sim.undeploy_module(service["app"], service["service_name"], service["des"])
            sim.stop_process(service["des"])
        else:
            sim.undeploy_module(service["app"], service["service_name"], service["clone_des"])
            sim.stop_process(service["clone_des"])
    except KeyError:
        pass


def get_free_space_on_nodes(sim):
    currentOccupation = dict(
        [a, int(x)] for a, x in nx.get_node_attributes(G=sim.topology.G, name="HwReqs").items())

    for app in sim.alloc_module:
        dict_module_node = sim.alloc_module[app]  # modules deployed
        for module in dict_module_node:
            for des in dict_module_node[module]:
                size = sim.get_size_service(app, sim.alloc_level[des])
                currentOccupation[sim.alloc_DES[des]] -= size

    return currentOccupation

def get_app_identifier(nameservice):
    return int(nameservice[0:nameservice.index("_")])

def get_nodes_with_users(routing):
    nodes_with_users = defaultdict(list)
    for (_,node_user,user_service) in routing.controlServices.keys():
        nodes_with_users[node_user].append(get_app_identifier(user_service))
    return nodes_with_users

def modify_service_level(sim, id_service, new_level):
    sim.alloc_level[id_service] = new_level


def maximun_Request_Level_byApp(apps_level, app):
    flavours = apps_level[app]
    f = [flavours[k][1] for k in flavours]
    ix = f.index(max(f))
    return (f, list(flavours.keys())[ix])


def computeFitness(apps_level, current_service, operation, Ostatus, Fstatus):
    # print("*" * 10)
    # print(Ostatus)
    # print(operation)
    # print(Fstatus)
    # print(current_service)
    # print(apps_level)


    if operation == "undeploy":
        if "DifReqChannels" in Ostatus["requests"]:
            if Ostatus["requests"]["DifReqChannels"] > 0:  # there are requests and the service goes out
                return 0.0
            if Ostatus["requests"]["DifReqChannels"] == 0:
                return 1.0
        else:
            return 1.0 #There was not requests

    # After undeploy operation, any other operation it's negative whether the node HW utilization >1
    if Fstatus["node"]["Utilization"]>1:
        return 0.0

    # There are not requests
    # The only viable operation is UNDEPLOY
    if "SumLat" not in Ostatus["requests"]:
        return 0.0


    if operation == "migrate":
        if "NextSumLat" in Fstatus["requests"]:
            if Ostatus["requests"]["DifReqChannels"] == 1:
                if Ostatus["requests"]["SumLat"] > Fstatus["requests"]["NextSumLat"]:
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0

    if operation == "replicate":
        if Ostatus["requests"]["DifReqChannels"] > 1 and current_service["SupportedRequests"]<Ostatus["requests"]["SumRequests"]:
             if "SumLat" not in Fstatus["requests"]: #The current service not receive requests
                return 0.0
             else:
                return 1.0
        else:
            # Si la utilizacion es mayor que 1, pero solo hay un canal es mejor una migración
            return 0.0




    if operation in FLAVOURS:  # adapt to small
        # return adapt_fitness_v1(apps_level,current_service, Ostatus, Fstatus)
        return adapt_fitness_v2(apps_level, current_service, Ostatus, Fstatus)


def adapt_fitness_v1(apps_level, current_service, Ostatus, Fstatus):
    list_supported_requests, max_flavour_name = maximun_Request_Level_byApp(apps_level, current_service["app"])
    current_requests = Fstatus["requests"]["SumRequests"]

    supported_ORequests = apps_level[current_service["app"]][current_service["old_level"]][1]
    supported_FRequests = apps_level[current_service["app"]][current_service["level"]][1]

    old_flavour = current_service["old_level"]
    future_flavour = current_service["level"]

    if current_requests > max(list_supported_requests) and old_flavour == max_flavour_name:
        # print("Impossible to improve better")
        return 0.0

    if current_requests > max(list_supported_requests) and future_flavour == max_flavour_name:
        # print("Best adapt flavour possible")
        return 1.0
        # print("DONE----")

    predicate = lambda x: x >= current_requests
    item = next(filter(predicate, list_supported_requests), None)
    ix_future_flavour = list_supported_requests.index(supported_FRequests)
    ix_current_flavour = list_supported_requests.index(supported_ORequests)

    if item is not None:  # There is a level that supports the current flow of requests
        ix_best_flavour = list_supported_requests.index(item)
        if ix_future_flavour == ix_best_flavour:
            # print("Best movement")
            return 1.0
        if ix_future_flavour < ix_best_flavour:
            # print("Worst movement")
            return 0.0

        if ix_future_flavour > ix_best_flavour:
            # print("It's good but it can be better")
            numberFlavours = len(list_supported_requests)
            return (numberFlavours - ix_future_flavour + ix_best_flavour) / numberFlavours

    if item is None:
        # we need to look for the best, we need the last
        # ix_best_flavour = len(list_supported_requests) - 1
        return ((ix_future_flavour + 1) / len(list_supported_requests))  # distance to the best one

def adapt_fitness_v2(apps_level, current_service, Ostatus, Fstatus):
    list_supported_requests, max_flavour_name = maximun_Request_Level_byApp(apps_level, current_service["app"])

    if len(Fstatus["requests"])==0: #Empty request
        return 0.0 # An adapt operation when there are not requests

    current_requests = Fstatus["requests"]["SumRequests"]
    supported_FRequests = apps_level[current_service["app"]][current_service["level"]][1]

    rate =  current_requests / float(supported_FRequests)
    predicate = lambda x: x >= current_requests
    item = next(filter(predicate, list_supported_requests), None)

    if item is not None:
        if supported_FRequests > item:  # This adaptation is over-dimensioned
            return rate
        else:
            return 1.0
    else:
        return rate

class BowserManager():
    def __init__(self, path_csv_traces, path_results, apps_level, samples_file, iteration):
        self.logger = logging.getLogger(__name__)
        self.apps_level = apps_level
        self.all_deployed_services = None  # list of the deployed services
        self.ixService = -1
        self.current_service = None
        self.current_ioperation = 0
        self.stats_cache = dict()
        self.available_HW_on_nodes = dict()
        # self.baseline_CSVpointer = None  # CSV pointer to load simulator traces #v1
        self.last_CSVpointer = 0
        self.path_csv_traces = path_csv_traces
        self.do_time = 0  # v2 - we analyse all normal-periods of the simulation, where the situation of the services is equal to the initial
        # if requests follow an exponential distribution the periods should be more large or may be clean all the simulator states :S
        # Works "well" with a deterministic distribution
        # STATS
        self.iteration = iteration
        self.path_results = path_results
        self.sample_state = ""
        self.number_samples = 0

        # Store stats, the simulation samples
        self.samples_file = samples_file
        # with open(self.path_results + self.samples_file, "w") as f:
        #     columns = ",".join([item for item in MEASURES])
        #     f.write(columns + "\n")



    def get_latency(self, path, topology):
        speed = 0
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            speed += topology.G.edges[link][Topology.LINK_PR]
        return speed

    def __call__(self, sim, routing, case_folder):
        """
        This functions is called periodically by the simulator

        It has three states:
             --- meanwhile simulation runs
             - s0: choose a service and pass to (s1)
             - s1: DO one operation and pass to (s2)
             --- meanwhile simulation runs
             - s2: UNDO each operation and go to next operation (s1), if finish pass to (s0)

        :param sim:
        :param routing:
        :param case_folder:
        :return:
        """
        # print("\nBOWSER IS HERE: Number of Service Managed: ", (self.ixService + 1))
        # print("*" * 30)
        # sim.print_debug_assignaments()
        # Next if-block runs one time:
        # We get the list of deployed services in the infrastructure &
        # We get the list of available resources in the nodes.
        if self.all_deployed_services is None:
            id = 0
            self.all_deployed_services = []
            for app in sim.alloc_module:
                for service in sim.alloc_module[app]:
                    for des in sim.alloc_module[app][service]:
                        self.all_deployed_services.append(
                            {"id": id, "app": app, "des": des, "service_name": service, "node": sim.alloc_DES[des],
                             "old_level": sim.alloc_level[des], "level": sim.alloc_level[des],
                             "SupportedRequests": self.apps_level[app][sim.alloc_level[des]][1]
                             })
                        id += 1

            self.degree_centrality = nx.degree_centrality(sim.topology.G)
            self.node_with_users = get_nodes_with_users(routing)

        # State 0
        if self.current_service is None:
            self.ixService += 1
            self.current_ioperation = 0
            # if self.ixService == 1:  # len(self.all_deployed_services): #TODO Swap len() to int value for DEBUG -ing
            if self.ixService == len(self.all_deployed_services): #TODO Swap len() to int value for DEBUG -ing
                sim.stop = True
                return None

            # print(self.data_placements[self.ixService])
            self.current_service = self.all_deployed_services[self.ixService]

            self.state = "DO"

            # if self.baseline_CSVpointer is None:  # We do only one time after a execution period
            #     sim.metrics.flush()
            #     df = pd.read_csv(self.path_csv_traces + ".csv")
            #     self.baseline_CSVpointer = len(df)
            #     self.last_CSVpointer = self.baseline_CSVpointer
            #     # print("Size of the baseline region: ", self.baseline_CSVpointer)
            #     # print(df)

        # State 1
        if self.state == "DO":
            operation = OPERATIONS[self.current_ioperation]

            # Control adapt operation related with flavours and service's flavours
            search_valid_adapt_operation = operation in FLAVOURS  # If the operation is adapt-type
            while search_valid_adapt_operation:
                if not operation in self.apps_level[self.current_service["app"]]:
                    self.current_ioperation += 1
                else:
                    if self.current_service["level"] == operation:  # jump this operation
                        self.current_ioperation += 1
                    else:
                        search_valid_adapt_operation = False
                        break

                if self.current_ioperation == len(OPERATIONS):  # Not more operations
                    self.current_service = None
                    return  # Jump to another state -> s0

                operation = OPERATIONS[self.current_ioperation]  # next available operation
            # end while
            # We measure the performance status of the service in the current situation
            status = self.get_service_status(sim, routing, self.current_service)
            self.Ostatus = status
            self.storeStatus(status, operation, self.current_service, state="DO")

            # print("\n\nDO\n",status)
            # print(operation)
            # print(self.current_service)
            # print("DO.tIME ",sim.env.now)

            # We will only consider requests from now on in the trace file.
            self.do_time = sim.env.now
            # We perform the action
            self.action_done = self.do_action(sim, operation, self.current_service)

            # sim.print_debug_assignaments()

            self.state = "UNDO"  # After apply, we undo the previous operation

        # State 2
        else:
            # We undo the operation
            if self.action_done:
                operation = OPERATIONS[self.current_ioperation]


                Fstatus = self.get_service_status(sim, routing, self.current_service, empty=(operation == "undeploy"))

                # we compute the fitness
                valueFit = computeFitness(self.apps_level, self.current_service, operation, self.Ostatus, Fstatus)

                self.storeStatus(Fstatus, operation, self.current_service, state="UNDO", fit=valueFit)

                # print("UNDO\n",status)
                # print(self.current_service)
                # print("Undo TIME ", sim.env.now)
                self.undo_action(sim, operation, self.current_service)
                # sim.print_debug_assignaments()

            self.current_ioperation += 1
            self.state = "DO"

            if self.current_ioperation == len(OPERATIONS):
                self.current_service = None

    def storeStatus(self, status, operation, current_service, state, fit=None):
        """

        :param status:
        :param operation:
        :param current_service:
        :param state:
        :return:
        """
        if state == "DO": # Pre-Action Data
            self.sample_state = str(self.iteration)+","+ operation + "," + str(current_service["app"]) + "," + current_service["old_level"] + ","  #CTS FACTS

            SReq  = self.apps_level[current_service["app"]][current_service["old_level"]][1]
            HWreq = self.apps_level[current_service["app"]][current_service["old_level"]][0]
            self.sample_state += str(SReq) + "," + str(HWreq) + ","
        else:
            SReq = self.apps_level[current_service["app"]][current_service["level"]][1]
            HWreq = self.apps_level[current_service["app"]][current_service["level"]][0]

            if operation != "undeploy":
                # self.sample_state += "NN,"

                self.sample_state += str(self.current_service["DSTNode"]["DstHWUsed"])+ ","
                self.sample_state += str(self.current_service["DSTNode"]["DstUtilization"])+ ","

                # self.sample_state +="FNN,"
            else:
                self.sample_state += ("0," * 2)

            self.sample_state += str(SReq) + "," + str(HWreq) + ","



        # self.sample_state += "IR,"
        attrs = status["requests"]
        for key in attrs:
            self.sample_state += str(attrs[key]) + ","
        if len(attrs) == 0:
            self.sample_state += ("0," * 6)

        # self.sample_state += "FR,"

        # self.sample_state += "IN,"


        attrs = status["node"]
        for key in attrs:
            self.sample_state += str(attrs[key]) + ","
        # self.sample_state += "FN,"
        # TODO include more measures :colums


        # We include the fit
        if fit is not None:
            self.sample_state += "%f," % fit

        # we undo the operation, we can store all the measures of a action, old_state, future_state, & fit
        if state == "UNDO":
            self.sample_state = self.sample_state[:-1]
            with open(self.path_results + self.samples_file, "a") as f:
                f.write(self.sample_state + "\n")
                self.number_samples += 1

    def get_service_status(self, sim, routing, service, empty=False):
        """
            Get performance indicators of the service

            There are several CSV parts regarding with the states.

            PERIOD.0: Simulation.time.0 .. Simulation.time.first-DOAction
                It is the baseline of all services

            PERIOD odd: Simulation.time.DoAction .. Simulation.time.UndoAction
                It the period where the action affect the measures

            PERIOD pair: Simulation.time.UndoAction .. Simulation.time.DoAction
                We ignore this period. If the period is enough big and the system is stable.
                This period will be the same than the first
                PERIOD.pai == PERIOD.0 + simulation.time

        """
        sim.metrics.flush()
        df = pd.read_csv(self.path_csv_traces + ".csv")

        df2 = df[self.last_CSVpointer:]
        df2 = df2[df2["time_emit"] >= self.do_time]
        self.last_CSVpointer = len(df)

        listOfRequests = self.__get_requests(sim, routing, service, df2, empty)

        nodeInfo = self.__get_node_info(sim,routing,service)

        # TODO Include more metrics :columsa´´´´´´´çç
        stats = {"requests": listOfRequests,"node":nodeInfo}

        return stats

    def do_action(self, sim, operation, service):
        self.available_HW_on_nodes = get_free_space_on_nodes(sim)
        currentOccupation = dict(
            [a, int(x)] for a, x in nx.get_node_attributes(G=sim.topology.G, name="HwReqs").items())

        if operation == "undeploy":
            # undeploy_module(sim, service) # We can ignore this operation since the State without this service is always all zeros.
            return True

        if operation in FLAVOURS:
            self.current_service["level"] = operation

            target_node = self.current_service["node"]
            total_space = currentOccupation[target_node]
            free_space = self.available_HW_on_nodes[target_node]
            self.current_service["DSTNode"] = {}
            # self.current_service["DSTNode"]["DstHWTotal"] = total_space
            self.current_service["DSTNode"]["DstNode"] = target_node
            self.current_service["DSTNode"]["DstHWUsed"] = (total_space - free_space)
            self.current_service["DSTNode"]["DstUtilization"] = (total_space - free_space) / total_space
            # self.current_service["DSTNode"]["DstDegree"] = sim.topology.G.degree(target_node)
            # self.current_service["DSTNode"]["DstCentrality"] = self.degree_centrality[target_node]
            # self.current_service["DSTNode"]["DstConnectedUsers"] = len(self.node_with_users[target_node])
            modify_service_level(sim, self.current_service["des"], operation)
            return True

        if operation == "migrate":
            # We get neighbours
            neighs = list(sim.topology.G.neighbors(self.current_service["node"]))
            # From that neighbours we filter the nodes with HW availability
            # service_HW = self.apps_level[self.current_service["app"]][self.current_service["level"]][0]
            # available_neighs = [n for n in neighs if self.available_HW_on_nodes[n] - service_HW >= 0]

            # print("OPERATION MIGRATE")
            # print(neighs)

            if len(neighs) > 0:
                target_node = np.random.choice(neighs, 1)[0]


                total_space = currentOccupation[target_node]
                free_space =  self.available_HW_on_nodes[target_node]
                self.current_service["DSTNode"] = {}
                self.current_service["DSTNode"]["DstNode"] = target_node
                self.current_service["DSTNode"]["DstHWUsed"] = (total_space-free_space)
                self.current_service["DSTNode"]["DstUtilization"] = (total_space-free_space) / total_space

                undeploy_module(sim, service)

                self.current_service["old_node"] = self.current_service["node"]
                self.current_service["node"] = target_node

                deploy_module(sim, service, node_dst=target_node)

                return True
            else:
                self.logger.warning("There are not available neighbours for migrate operation")
                return False

        if operation == "replicate":
            # We get neighbours
            neighs = list(sim.topology.G.neighbors(self.current_service["node"]))

            # From that neighbours we filter the nodes with HW availability
            # HWservice_HW = self.apps_level[self.current_service["app"]][self.current_service["level"]][0]
            # available_neighs = [n for n in neighs if self.available_HW_on_nodes[n] - service_HW >= 0]


            # print("OPERATION REPLICATE")
            # print(neighs)

            if len(neighs) > 0:
                target_node = np.random.choice(neighs, 1)[0]

                total_space = currentOccupation[target_node]
                free_space =  self.available_HW_on_nodes[target_node]
                self.current_service["DSTNode"] = {}
                # self.current_service["DSTNode"]["DstHWTotal"] = total_space

                self.current_service["DSTNode"]["DstNode"] = target_node
                self.current_service["DSTNode"]["DstHWUsed"] = (total_space-free_space)
                self.current_service["DSTNode"]["DstUtilization"] = (total_space-free_space) / total_space
                # self.current_service["DSTNode"]["DstDegree"] = sim.topology.G.degree(target_node)
                # self.current_service["DSTNode"]["DstCentrality"] = self.degree_centrality[target_node]
                # self.current_service["DSTNode"]["DstConnectedUsers"] = len(self.node_with_users[target_node])


                self.current_service["to_node"] = target_node
                deploy_module(sim, service, node_dst=target_node, replicate=True)
                return True
            else:
                self.logger.warning("There are not available neighbours for replicate operation")
                return False

    def undo_action(self, sim, operation, service):
        if operation == "undeploy":
            # deploy_module(sim, service)
            pass
        if operation in FLAVOURS:
            self.current_service["level"] = self.current_service["old_level"]
            modify_service_level(sim, self.current_service["des"], self.current_service["old_level"])
        if operation == "migrate":
            undeploy_module(sim, service)
            deploy_module(sim, service, node_dst=service["old_node"])  # the first assigment
        if operation == "replicate":
            undeploy_module(sim, service, replicate=True)

    def __get_requests(self, sim, routing, service, df, empty=False):
        if empty:
            return {"DifReqChannels": 0,
                    "TotReqChannels": 0,
                    "SumRequests": 0,
                    "AvgRequests": 0,
                    "SumLat": 0,
                    "AvgLat": 0}

        # df.to_csv("debug1.csv") # DEBUG
        df = df[df["DES.dst"] == service["des"]]
        latencies = []
        requests = []
        incomingNodes = set()

        if len(df) > 0:
            # WARNING: This code could generate bugs IF the topology is dynamic (ie. node failures)
            dg = df.groupby("TOPO.src")["TOPO.src"].count()
            for node_src, sumMessages in dg.iteritems():
                path = routing.get_path_from_src_dst(sim, node_src, service["node"])
                latency = self.get_latency(path, sim.topology)
                latencies.append(latency)
                requests.append(sumMessages)

                if sumMessages > 0:  # No sure about this if-statement
                    if len(path) == 1:
                        node_code = "self"
                    else:
                        node_code = path[-2]
                    incomingNodes.add(node_code)

            return {"DifReqChannels": len(incomingNodes),
                    "TotReqChannels": len(requests),
                    "SumRequests": np.sum(requests),
                    "AvgRequests": np.mean(requests),
                    "SumLat": np.sum(latencies),
                    "AvgLat": np.mean(latencies)}
        else:
            self.logger.warning("WARN - There are not new messages among users and service")
            return {}

    def __get_node_info(self,sim,routing,service):

        self.available_HW_on_nodes = get_free_space_on_nodes(sim)  # double check, but there're always space due to rules

        des = service["des"]
        # if "clone_des" in service:  #In case of replication...
        #     des = service["clone_des"]
        node = sim.alloc_DES[des]

        degree = sim.topology.G.degree(node)
        HWtotal = float(sim.topology.G.nodes[node]["HwReqs"])
        HWused = self.available_HW_on_nodes[node]
        centrality = self.degree_centrality[node]
        nUsers = len(self.node_with_users[node])

        data = {}
        # data["Node"] = node
        data["HWTotal"] = HWtotal
        data["HWUsed"] = HWused
        data["Utilization"] = (HWtotal-HWused)/HWtotal
        data["Degree"] = degree
        data["Centrality"] = centrality
        data["ConnectedUsers"] = nUsers

        return data

