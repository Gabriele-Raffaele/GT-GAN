# ==== MODIFICHE AL TUO CODICE ESISTENTE ====

# 1. SOSTITUISCI la tua classe ContinuousGenerator con questa versione migliorata:

class ContinuousGenerator(nn.Module):
    """
    Generator che mantiene stato nascosto tra finestre consecutive
    """
    def __init__(self, args, base_generator, hidden_size):
        super().__init__()
        self.args = args
        self.base_generator = base_generator
        self.hidden_size = hidden_size
        
        # LSTM per propagazione stato tra finestre
        self.state_propagator = nn.LSTM(
            hidden_size, hidden_size, 
            batch_first=True, num_layers=1
        )
        
        # Proiezioni per integrare stato nascosto
        self.state_to_noise = nn.Linear(hidden_size, args.effective_shape)
        self.noise_combiner = nn.Linear(args.effective_shape * 2, args.effective_shape)
        
        # Stato globale che persiste tra chiamate (NUOVO)
        self.global_hidden = None
        self.global_cell = None
        
    def reset_state(self, batch_size=None):
        """Reset dello stato nascosto"""
        self.global_hidden = None
        self.global_cell = None
        
    def forward_single_window(self, z_single, times_single, device, z=True, use_global_state=True):
        """
        NUOVO: Forward per una singola finestra, mantenendo stato globale
        """
        batch_size = z_single.size(0)
        
        if z:  # Modalità generazione
            if use_global_state and self.global_hidden is not None:
                # Usa stato globale dalla finestra precedente
                if self.global_hidden.size(1) != batch_size:
                    # Adatta dimensione se necessario
                    self.global_hidden = self.global_hidden[:, :batch_size, :].contiguous()
                    self.global_cell = self.global_cell[:, :batch_size, :].contiguous()
                
                # Genera influenza dallo stato precedente
                state_input = self.global_hidden.squeeze(0)  # [batch_size, hidden_size]
                state_noise = self.state_to_noise(state_input)
                
                # Combina con noise originale
                combined_noise = self.noise_combiner(
                    torch.cat([z_single[:, 0, :], state_noise], dim=1)
                )
                
                z_modified = z_single.clone()
                z_modified[:, 0, :] = combined_noise
            else:
                z_modified = z_single
                
            # Genera output
            output = run_model(self.args, self.base_generator, z_modified, times_single, device, z=True)
            
            if use_global_state:
                # Aggiorna stato globale
                last_output = output[:, -1:, :]  # [batch_size, 1, hidden_size]
                
                if self.global_hidden is None:
                    # Prima finestra - inizializza stato
                    h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
                    c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
                    prev_state = (h0, c0)
                else:
                    prev_state = (self.global_hidden, self.global_cell)
                
                _, (new_hidden, new_cell) = self.state_propagator(last_output, prev_state)
                
                # Salva nuovo stato globale (detached)
                self.global_hidden = new_hidden.detach()
                self.global_cell = new_cell.detach()
            
            return output
        else:
            # Modalità supervisione - usa direttamente base generator
            if self.args.kinetic_energy is None:
                return run_model(self.args, self.base_generator, z_single, times_single, device, z=False)
            else:
                return run_model(self.args, self.base_generator, z_single, times_single, device, z=False)
    
    def forward(self, z_or_h, times, device, z=True, propagate_state=True):
        """
        Mantiene compatibilità con il codice esistente
        """
        if propagate_state:
            # NUOVO: Se propagate_state=True, processa sequenzialmente
            batch_size = z_or_h.size(0)
            all_outputs = []
            
            for i in range(batch_size):
                single_input = z_or_h[i:i+1]
                single_times = times[i:i+1]
                
                if z:
                    output = self.forward_single_window(single_input, single_times, device, z=True, use_global_state=True)
                else:
                    output = self.forward_single_window(single_input, single_times, device, z=False, use_global_state=False)
                
                all_outputs.append(output)
            
            # Ricostruisci batch
            if z:
                return torch.cat(all_outputs, dim=0)
            else:
                if self.args.kinetic_energy is None:
                    losses = [out[0] for out in all_outputs]
                    return torch.mean(torch.stack(losses)), None
                else:
                    losses_s = [out[0] for out in all_outputs]
                    losses = [out[1] for out in all_outputs]  
                    reg_states = [out[2] for out in all_outputs]
                    return torch.mean(torch.stack(losses_s)), torch.mean(torch.stack(losses)), torch.mean(torch.stack(reg_states))
        else:
            # Modalità compatibilità - usa implementazione originale
            batch_size = z_or_h.size(0)
            
            if z:  # Modalità generazione con noise
                z = z_or_h
                output = run_model(self.args, self.base_generator, z, times, device, z=True)
                return output
            else:  # Modalità supervisione con hidden states
                h = z_or_h
                if self.args.kinetic_energy is None:
                    loss_s, loss = run_model(self.args, self.base_generator, h, times, device, z=False)
                    return loss_s, loss
                else:
                    loss_s, loss, reg_state = run_model(self.args, self.base_generator, h, times, device, z=False)
                    return loss_s, loss, reg_state


# 2. MODIFICA la funzione train() - sostituisci solo queste righe:

# TROVA questa riga nel tuo codice:
# generator = ContinuousGenerator(args, generator, hidden_size=24).to(device)

# SOSTITUISCILA con:
# base_generator = generator  # Salva riferimento al generator originale
# generator = ContinuousGenerator(args, base_generator, hidden_size=24).to(device)


# 3. AGGIUNGI questa logica di reset stato nel loop di training:

# TROVA questa parte nel tuo codice:
# for step in tqdm(range(1, max_steps+1)):
#     h_prev = None
#     # Reset stato del generator all'inizio di ogni epoch
#     if (step - 1) % num_batches_per_epoch == 0:
#         generator.reset_state()
#         print(f"Reset generator state at step {step}")

# SOSTITUISCILA con:
"""
for step in tqdm(range(1, max_steps+1)):
    h_prev = None
    # Reset stato del generator all'inizio di ogni epoch
    if (step - 1) % num_batches_per_epoch == 0:
        generator.reset_state()
        print(f"Reset generator state at step {step}")
"""


# 4. AGGIUNGI parametro di controllo sequenzialità:

# Nel parser degli argomenti, AGGIUNGI:
# parser.add_argument("--sequential_generation", action="store_true", default=True, 
#                     help="Enable sequential generation between windows")

# Poi nel training, MODIFICA queste chiamate:

# TROVA:
# h_hat = generator(z, times, device, z=True, propagate_state=True)

# SOSTITUISCI con:
# h_hat = generator(z, times, device, z=True, propagate_state=args.sequential_generation)

# TROVA:
# loss_s, loss = generator(h, times, device, z=False, propagate_state=False)

# SOSTITUISCI con:  
# loss_s, loss = generator(h, times, device, z=False, propagate_state=False)  # Mantieni False per supervisione


# 5. MODIFICA la parte di inferenza:

# TROVA questa sezione nel blocco else (modalità inferenza):
# with torch.no_grad():
#     # ... setup ...
#     h_hat = run_model(args, generator, z, times, device, z=True)

# SOSTITUISCILA con:
"""
with torch.no_grad():
    # Crea il ContinuousGenerator per inferenza
    path = here / 'dumarey_model/dumarey_pretrained'
    
    # Carica il generator base prima
    base_generator = build_model_tabular_nonlinear(
        args, args.effective_shape, regularization_fns=regularization_fns,).to(device)
    set_cnf_options(args, base_generator)
    
    # Crea ContinuousGenerator e carica stato
    continuous_gen = ContinuousGenerator(args, base_generator, hidden_size=24).to(device)
    continuous_gen.load_state_dict(torch.load(path / "generator.pt", map_location=device))
    continuous_gen.reset_state()  # Reset per inferenza
    
    # ... resto del setup ...
    
    # Genera con sequenzialità
    h_hat = continuous_gen(z, times, device, z=True, propagate_state=True)
"""


# 6. AGGIUNGI funzione di utilità per debug:

def debug_sequential_generation(generator, dataset, args, device, num_windows=5):
    """
    Funzione per verificare che la sequenzialità funzioni correttamente
    """
    generator.reset_state()
    generator.eval()
    
    print("=== DEBUG: Sequential Generation Test ===")
    
    with torch.no_grad():
        for i in range(min(num_windows, len(dataset))):
            batch = dataset[(i, 1)]  # Una finestra alla volta
            x = batch['data'].to(device)
            obs = x[:, :, -1]
            
            time = torch.FloatTensor(list(range(24))).to(device)
            times = time.unsqueeze(0).unsqueeze(2)
            
            z = torch.randn(1, x.size(1), args.effective_shape).to(device)
            
            # Genera con stato
            h_hat = generator(z, times, device, z=True, propagate_state=True)
            
            print(f"Finestra {i}: Shape output = {h_hat.shape}")
            print(f"  - Primo valore: {h_hat[0, 0, 0].item():.4f}")
            print(f"  - Ultimo valore: {h_hat[0, -1, 0].item():.4f}")
            print(f"  - Stato globale presente: {generator.global_hidden is not None}")
    
    print("=== Fine Debug ===")


# 7. CHIAMATA DI DEBUG (opzionale):

# Aggiungi questa chiamata dopo il training o prima dell'inferenza:
# if args.train and step == max_steps:  # Alla fine del training
#     debug_sequential_generation(generator, dataset, args, device)