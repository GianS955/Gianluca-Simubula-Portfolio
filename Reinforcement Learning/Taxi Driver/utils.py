import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

@dataclass
class Cell:
    i: int
    j: int
    type: str
    obstacles:list[str]
    coordinates = []
    color = 'k'

    def __post_init__ (self):
        with open('env.json','r') as file:
            env = json.load(file)

            cell = [c for c in env['cells'] if c['type']== self.type]
            if (len(cell)==0):
                raise Exception(f"No definitions present for cells with type {self.type}")
            
            obstacles = self.obstacles
            self.obstacles = []
            for obstacle in obstacles:
                corr = [Obstacle(**o) for o in env['obstacles'] if o["type"] == obstacle]
                if (len(corr)==1):
                    self.obstacles.append(corr[0])
            
            self.color = cell[0]["color"]
            self.coordinates = [self.i,self.j]

@dataclass
class Obstacle:
    type:str
    row: int
    column: int

@dataclass
class Action:
    type: str
    row: int
    column: int
    passenger: int
    reward: list[int]
    annotation: str

@dataclass
class Passenger:
    position: int
    destination:list[int]

    def __post_init__(self):
        self.on_board = False
        self.is_active = True

@dataclass
class Grid:
    rows: int
    columns : int
    cells: list[Cell]

    def get_neighbour(self,cell,action):
        neighbour = self.get_cell_by_coord([cell.i + action.row, cell.j +action.column])

        if (neighbour is None):
            return neighbour
        
        if (neighbour.type == 'building'):
            return None
        
        
        for o in cell.obstacles:
            match o.type:
                case "N":
                    if(cell.i>neighbour.i):
                        return None
                case "S":
                    if (cell.i<neighbour.i):
                        return None
                case "W":
                    if (neighbour.j<cell.j):
                        return None
                case "E":
                    if (neighbour.j>cell.j):
                        return None
                        
        for o in neighbour.obstacles:
            match o.type:
                case "N":
                    if(cell.i<neighbour.i):
                        return None
                case "S":
                    if (cell.i>neighbour.i):
                        return None
                case "W":
                    if (neighbour.j>cell.j):
                        return None
                case "E":
                    if (neighbour.j<cell.j):
                        return None
        
        return neighbour

    def get_cell_by_coord(self, coordinates):
        try: 
            cell = [cell for cell in self.cells if cell.i == coordinates[0] and cell.j == coordinates[1]][0]
        except:
            cell = None
        return cell 

    def get_pick_up_cell(self,index=None):
        cells = [c for c in self.cells if c.type== "pick-up/drop-off"]
        if (index is None):
            return cells
        else:
            try:
                return cells[index]
            except:
                return None

class Agent:
    def __init__(self,state,ax):
        self.icon = 'images\\icon.png'
        self.state = state
        self.marker = ax.plot(self.state[0], self.state[1], 'ro', markersize=12)

    def update(self,action,ax):
        self.state = (self.state[0] + action.row, self.state[1]+action.column, self.state[2]+action.passenger)
        self.marker = ax.plot(self.state[0], self.state[1], 'ro', markersize=12)

def load_grid(file):
    with open(file,'r') as f:
        data = json.load(f)
        cells = [Cell(**c) for c in data['cells']]
        grid = Grid(
            rows=data["rows"],
            columns=data["columns"],
            cells=cells
        )
        return grid

class Gym:

    actions = []

    def __init__(self, file):
        self.grid = load_grid(file)
        self.actions = self.load_actions()
        self.passengers = self.load_passengers(file)
        self.policy = {}
        self.values = {}

    def load_actions(self):
        with open('env.json','r') as file:
            env = json.load(file)
            actions = [Action(**a) for a in env["actions"]]
            return actions
    
    def load_passengers(self,file):
        with open(file,'r') as f:
            data = json.load(f)
            passengers = [Passenger(**p) for p in data["passengers"]]
            return passengers

    def initialize(self):
        for passenger_location in [0,1,2,np.inf]: # np.inf is on board state
            for passenger_destination in [0,1,2]:
                for cell in self.grid.cells:
                    self.values[(cell.i,cell.j,passenger_location,passenger_destination)] = 0
        
    def pick_up_drop_off(self, cell,result):
        for p in [p for p in self.passengers if p.is_active]:
            if (result):
                if (p.i == cell.coordinates[0] and p.j == cell.coordinates[1]):
                    p.on_board = result
            else:
                if (p.destination[0] == cell.coordinates[0] and p.destination[1] == cell.coordinates[1]):
                    p.on_board = False
                    p.is_active = False

    def has_cell_passenger(self,cell):
        return len([p for p in self.passengers if p.i == cell.coordinates[0] and p.j == cell.coordinates[1] and p.is_active]) ==1   

    def get_on_board_passenger(self):
        for p in self.passengers:
            if p.on_board:
                return p

    def value_iteration(self, max_iteration, treshold, gamma):
        import time
        iteration = 0        
        while(True):
            start = time.perf_counter()
            V = {}  
            deltas = {}
            for state in self.values.keys():
                if (state == (4,3,2,1)):
                    a = 0
                cell = self.grid.get_cell_by_coord([state[0],state[1]])
                passenger_location = self.grid.get_pick_up_cell(state[2])
                passenger_destination = self.grid.get_pick_up_cell(state[3])

                values = {} 
                # terminal state:
                if (state[2]==state[3]):
                    deltas[state] = 0
                    self.policy[state] = []
                    V[state] = 0
                    continue

                for action in self.actions:
                    match action.type:
                        case "N"|"S"|"W"|"E":
                            neighbour = self.grid.get_neighbour(cell,action)
                            if neighbour:                                
                                values[action.type] = action.reward[0] + gamma * self.values[(neighbour.i,neighbour.j,state[2],state[3])]
                            else:
                                values[action.type] = action.reward[1] + gamma * self.values[state]

                        
                        case "pick-up":
                            if (passenger_location is not None):
                                if (cell.type == "pick-up/drop-off" and cell.coordinates == passenger_location.coordinates):
                                    values[action.type] = action.reward[0] + gamma * self.values[(cell.i,cell.j,np.inf,state[3])]
                            
                        case "drop-off":
                            if (passenger_location is None and cell.type == "pick-up/drop-off"):
                                if (passenger_destination.coordinates != cell.coordinates):
                                    values[action.type] = action.reward[1] + gamma * self.values[(cell.i,cell.j,state[3],state[3])]
                                else:
                                    values[action.type] = action.reward[0] +  gamma * self.values[(cell.i,cell.j,state[3],state[3])]                

                if not values:
                    V[state] = -np.inf
                    deltas[state] = 0
                else:    
                    max_value = np.max(list(values.values()))
                    V[state] = max_value
                    deltas[state] = abs(self.values[state] - max_value)
                    max_types = [k for k in values.keys() if values[k] == max_value]
                    actions = [a.type for a in self.actions if a.type in max_types]
                    self.policy[state] = actions    
                
            for state in V.keys():
                self.values[state] = V[state]                 

            iteration +=1
            if (np.max(list(deltas.values()))< treshold):
                end = time.perf_counter()
                print(f"Convergence reached in {iteration} iterations. [{np.round(end-start,3)} seconds elapsed]")
                break
            
            if (iteration > max_iteration):
                end = time.perf_counter()
                print(f"Max number of iterations reached.  [{np.round(end-start,3)} seconds elapsed]")
                break

    def play_episode(self, start = (0,0,2,1)):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython.display import HTML

        # building steps
        frames = []
        state = start
        frames.append([state[0]+0.5,state[1]+0.5])
        while(True):

            try:
                action = self.policy[state][0]
            except:
                break
            cell = self.grid.get_cell_by_coord([state[0],state[1]])
            passenger_location = self.grid.get_pick_up_cell(state[2])
            passenger_destination = self.grid.get_pick_up_cell(state[3])

            match action:
                case "N"|"S"|"W"|"E":
                    action = [a for a in self.actions if a.type == action][0]
                    neighbour = self.grid.get_neighbour(cell,action)
                    if (neighbour):                                
                        state = (neighbour.i,neighbour.j,state[2],state[3])
                        
                case "pick-up":
                    action = [a for a in self.actions if a.type == action][0]
                    if (state[2] !=np.inf):
                        if (cell.type == "pick-up/drop-off" and cell.coordinates == passenger_location.coordinates):
                            state = (cell.i,cell.j,np.inf,state[3])
                            
                case "drop-off":
                    action = [a for a in self.actions if a.type == action][0]
                    if (passenger_location is None and cell.type == "pick-up/drop-off"):                        
                        state = (cell.i,cell.j,state[3],state[3])       

            frames.append([state[1]+0.5,state[0]+0.5])
                
        fig, ax,img = self.plot_grid()

        title = ax.set_title("Step 0")
        agent_marker, = ax.plot([], [], 'o', markersize=15, color ='#C41E3A')
        def update(i):
            state = frames[i]
            agent_marker.set_data([state[0]],[state[1]])
            title.set_text(f"Step {i}")

            return (img, agent_marker,title)

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=400,
            blit=False
        )

        plt.close(fig)  # evita doppio output
        return HTML(anim.to_jshtml())

    def plot_grid(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        fig, ax = plt.subplots()
        heatmap = np.empty([self.grid.rows,self.grid.columns])
        annotations = {}
        legend = {}

        for cell in self.grid.cells:
            match cell.type:
                case "road":
                    val = 0
                case "building":
                    val = 1
                case _:
                    val =2
            heatmap[cell.i,cell.j] = val
            annotations[val] = cell.type
            legend[val] = cell.color
        
        legend = dict(sorted(legend.items()))
        annotations = dict(sorted(annotations.items()))

        sorted_rewards = sorted(set(annotations.keys()))
        sorted_colors = [legend_color for reward in sorted_rewards  for legend_color in [dict(zip(legend.keys(), legend.values()))[reward]]]

        # creiamo la colormap discreta
        cmap = ListedColormap(sorted_colors)
        norm = BoundaryNorm(sorted_rewards + [sorted_rewards[-1] + 1], cmap.N)

        img = sns.heatmap(heatmap, 
                    cmap=cmap,
                    norm = norm,
                    annot=False, cbar=False,square=True, linecolor='k', linewidths=0.05)

        for cell in self.grid.cells:
            for o in cell.obstacles:
                match (o.type):
                    case "N":
                        row = [cell.j,cell.j+1]
                        column = [cell.i-1, cell.i-1]
                    case "S":
                        row = [cell.j,cell.j+1]
                        column = [cell.i+1, cell.i+1]
                    case "W":
                        row = [cell.j,cell.j]
                        column = [cell.i+1, cell.i]
                    case "E":
                        row = [cell.j+1,cell.j]+1
                        column = [cell.i-1, cell.i]
                
                line = mlines.Line2D(
                    row,column,color = '#C41E3A', linewidth=2
                )
                ax.add_line(line)

        patches = []
        for value in annotations.keys():
            color = legend[value]
            name = annotations[value]
            patch = mpatches.Patch(color=color, label=name)
            patches.append(patch)

        ax.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            title="Cell Types"
            )
        
        return fig, ax, img
        

