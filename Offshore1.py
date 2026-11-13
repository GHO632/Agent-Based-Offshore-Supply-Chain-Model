import pandas as pd
import numpy as np
from mesa import Agent, Model
from datetime import timedelta
import os
from collections import deque


# WETTERDATEN LADEN UND VERARBEITEN


def load_weather_data():
    
    downloads_path = r"/home/Gregor.Hoffmann/"
    
    wave_df = pd.read_csv(os.path.join(downloads_path, 'waveheight.csv'))
    wave_df['time'] = pd.to_datetime(wave_df['time'])
    wave_df = wave_df[['time', 'wave_height']].copy()
    wave_df['date'] = wave_df['time'].dt.date
    

    wind_df = pd.read_csv(os.path.join(downloads_path, 'windspeed.csv'))
    wind_df['time'] = pd.to_datetime(wind_df['time'])
    wind_df['date'] = wind_df['time'].dt.date
    wind_daily = wind_df.groupby('date')['wind_speed'].max().reset_index()
    
    weather = wave_df[['date', 'wave_height']]
    weather = weather.merge(wind_daily, on='date', how='left')
    weather = weather.sort_values('date').reset_index(drop=True)
    weather['date'] = pd.to_datetime(weather['date'])
    
    
    return weather




# AGENTEN-KLASSEN


class MarketAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.inventory = 5          
        self.max_inventory = 50
        self.mean_weekly = 50.0
        self.internal_production_buffer = 0.0 
        

        self.state = "VERFÜGBAR"    
        self.order_queue = deque()  

    def sample_production_rate(self):   
        return np.random.lognormal(np.log(self.mean_weekly / 7), sigma=0.3)

    def receive_order(self, project_agent, amount):
        self.order_queue.append({'project': project_agent, 'amount': amount})

    def step(self):
        production_today = self.sample_production_rate()
        self.internal_production_buffer += production_today
        produced_turbines = int(self.internal_production_buffer)
        if produced_turbines > 0:
            allowable = min(produced_turbines, self.max_inventory - self.inventory)
            self.inventory += allowable
            self.internal_production_buffer -= allowable
        if self.inventory >= 0 and len(self.order_queue) == 0:
            self.state = "VERFÜGBAR"
        else:
            self.state = "AUSGELASTET"

        if self.inventory > 0 and self.order_queue:
            free_transports = [t for t in self.model.transport_agents if t.state == "FREI"]
            if free_transports:
                transport = free_transports[0]
                order = self.order_queue[0] 
                amount_to_ship = min(self.inventory, order['amount'], transport.transport_capacity)
                if amount_to_ship > 0:
                    self.inventory -= amount_to_ship
                    order['amount'] -= amount_to_ship
                    transport.assign_mission(target_project=order['project'], load=amount_to_ship)
                    if order['amount'] <= 0:
                        self.order_queue.popleft()


class TransportAgent(Agent):
    def __init__(self, model, unique_id):
        super().__init__(model)
        self.unique_id = unique_id
        self.transport_capacity = 5
        self.transport_duration = 3  
        self.turbines_loaded = 0
        self.target_project = None
        self.days_in_transit = 0
        self.state = "FREI"

    def can_transport(self):
        if self.model.current_weather is None:
            return False
        wave_height = self.model.current_weather['wave_height']
        windspeed = self.model.current_weather['wind_speed']
        return wave_height < 3.0 and windspeed < 20.0

    def assign_mission(self, target_project, load):
        self.target_project = target_project
        self.turbines_loaded = load
        self.state = "WARTESCHLANGE" 

    def step(self):
        if self.state == "FREI":
            return 

        elif self.state == "WARTESCHLANGE":
            if self.can_transport():
                self.state = "TRANSPORT_LÄUFT"
                self.days_in_transit = 0

        elif self.state == "TRANSPORT_LÄUFT":
            self.days_in_transit += 1
            if self.days_in_transit >= self.transport_duration:
                self.target_project.receive_turbines(self.turbines_loaded)
                
                self.turbines_loaded = 0
                self.target_project = None
                self.state = "FREI"


class ProjectAgent(Agent):
    def __init__(self, model, project_id):
        super().__init__(model)
        self.project_id = project_id
        self.turbines_needed = 50
        self.turbines_on_site = 0
        self.turbines_built = 0
        
        self.reorder_point = 5
        self.order_batch = 10
        self.open_orders = 0

        self.build_time_per_turbine = 2 
        self.current_build_progress = 0
        self.state = "PLANUNG"
        self.start_date = None
        self.completion_date = None

    def can_build(self):
        if self.model.current_weather is None:
            return False
        wave_height = self.model.current_weather['wave_height']
        windspeed = self.model.current_weather['wind_speed']
        return wave_height < 3.0 and windspeed < 15.0
    
    def receive_turbines(self, amount):
        self.turbines_on_site += amount
        self.open_orders -= amount

    def step(self):
        if self.state == "PLANUNG":
            self.start_date = self.model.current_day
            self.state = "WARTE_AUF_LIEFERUNG"
            self.model.market_agent.receive_order(self, self.order_batch)
            self.open_orders += self.order_batch
            return

        if self.state == "WARTE_AUF_LIEFERUNG":
            if self.turbines_on_site > 0:
                self.state = "BAU_AKTIV"

        if self.state == "BAU_AKTIV":
            if self.can_build():
                self.current_build_progress += 1
                if self.current_build_progress >= self.build_time_per_turbine:
                    self.turbines_on_site -= 1
                    self.turbines_built += 1
                    self.current_build_progress = 0
            if self.turbines_built >= self.turbines_needed:
                self.state = "PROJEKT_ABGESCHLOSSEN"
                self.completion_date = self.model.current_day
            elif self.turbines_on_site == 0:
                self.state = "WARTE_AUF_LIEFERUNG"
            
            remaining_need = self.turbines_needed - self.turbines_built
            effective_stock = self.turbines_on_site + self.open_orders

            if remaining_need > 0 and effective_stock <= self.reorder_point:
                batch = min(self.order_batch, remaining_need - effective_stock)
                if batch > 0:
                    self.model.market_agent.receive_order(self, batch)
                    self.open_orders += batch



# MODEL KLASSE


class OffshoreModel(Model):
    def __init__(self, weather_df, start_date_idx=0):
        super().__init__()
        self.weather_df = weather_df
        self.current_day_idx = start_date_idx
        self.current_day = weather_df.iloc[start_date_idx]['date']
        self.current_weather = None

        self.market_agent = MarketAgent(model=self)
        self.transport_agents = [TransportAgent(model=self, unique_id=f"Schiff_{i}") for i in range(2)] 
        self.project_agents = [ProjectAgent(model=self, project_id=i) for i in range(5)] 
        
        self.update_weather()
    
    def update_weather(self):
        idx = self.current_day_idx % len(self.weather_df)
        row = self.weather_df.iloc[idx]
        self.current_weather = {'wave_height': row['wave_height'],'wind_speed': row['wind_speed'],  'date': row['date']}

    def step(self):
        self.market_agent.step()
        for t in self.transport_agents:
            t.step()
        for p in self.project_agents:
            p.step()
        self.current_day_idx += 1
        self.current_day += timedelta(days=1)
        self.update_weather()

        return all(p.state == "PROJEKT_ABGESCHLOSSEN" for p in self.project_agents)
        
    def run_to_completion(self, max_steps=5000):
        for _ in range(max_steps):
            all_done = self.step()
            if all_done:
                return self.get_completion_times()
        return self.get_completion_times()

    def get_completion_times(self):
        times = []
        for project in self.project_agents:
            if project.completion_date is not None:
                days = (project.completion_date - project.start_date).days
                times.append(days)
            else:
                times.append(None)  
        return times



# MONTE-CARLO & SPEICHERN


def run_monte_carlo(num_runs=100): 
    weather_df = load_weather_data()
    max_completion_times = []
    
    for run_idx in range(num_runs):
        start_idx = np.random.randint(0, len(weather_df))
        model = OffshoreModel(weather_df, start_date_idx=start_idx)
        completion_times = model.run_to_completion()
        max_completion_times.append(max(completion_times))

    max_completion_times = np.array(max_completion_times)

    if len(max_completion_times) == 0:
        return None

    return {
        'completion_times': list(max_completion_times),
        'min': float(np.min(max_completion_times)),
        'max': float(np.max(max_completion_times)),
        'median': float(np.median(max_completion_times)),
        'mean': float(np.mean(max_completion_times)),
        'std': float(np.std(max_completion_times)),
        'percentile_90': float(np.percentile(max_completion_times, 90))
    }

def save_results(results, output_path):
    if not results:
        print("Keine Ergebnisse zum Speichern.")
        return
    df = pd.DataFrame({'completion_times': results['completion_times']})
    df.to_csv(output_path, index=False)
    print(f"\nErgebnisse gespeichert: {output_path}")
    print("\n" + "="*60)
    print("MONTE-CARLO SIMULATIONSERGEBNISSE")
    print("="*60)
    print(f"Min completion time:  {results['min']:.2f} Tage")
    print(f"Max completion time:  {results['max']:.2f} Tage")
    print(f"Mittelwert:           {results['mean']:.2f} Tage")
    print(f"Std. Abweichung:      {results['std']:.2f} Tage")
    print(f"90% Perzentil:        {results['percentile_90']:.2f} Tage")
    print("="*60)

if __name__ == '__main__':
    results = run_monte_carlo(num_runs=100)
    output_file = r'/home/Gregor.Hoffmann/monte_carlo_results.csv'
    save_results(results, output_file)