# Written by Sachin Venkatesh (the Hybrid U-NO pipeline this SFNO was built on); included with
# credit. Kevin Fortune claims no ownership of this file. See LICENSE.
import os
import sys
import numpy as np
from mpi4py import MPI
sys.path.append(os.getcwd()) 
import pp

# CONFIGURATION
DATA_DIR = "/path/to/5000run"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_forecast")
START_DUMP = 5000
END_DUMP = 5700
STRIDE = 1

# Strict H-AMR Config
pp.set_mpi(0)
pp.lowres1 = 1; pp.lowres2 = 1; pp.lowres3 = 1
pp.do_box = 0; pp.do_unigrid = 1; pp.axisym = 1
pp.check_files = 0; pp.interpolate_var = 0; pp.export_raytracing_RAZIEH = 0
pp.DISK_THICKNESS = 0.03; pp.RAD_M1 = 0; pp.RESISTIVE = 0; pp.TWO_T = 0; pp.P_NUM = 0

def save_static_grid(dump_id):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("Loading Grid/Metric...")
    pp.rblock_new(dump_id)
    pp.rpar_new(dump_id)
    pp.rgdump_direct()
    pp.rdump_griddata(DATA_DIR, dump_id)
    
    r, h, ph = pp.r[0], pp.h[0], pp.ph[0] #Nx, Ny, Nz
    gdet = pp.gdet[0] #Nx, Ny, Nz                   
    gcov = pp.gcov[:, :, 0, :, :, :] #(4, 4, Nx, Ny, Nz)
    gcon = pp.gcon[:, :, 0, :, :, :] #(4, 4, Nx, Ny, Nz)

    dx = np.array([pp._dx1, pp._dx2, pp._dx3]) #grid spacing

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "geometry.npz"),
        r=r, h=h, ph=ph, 
        gdet=gdet.astype(np.float32), gcov=gcov.astype(np.float32), gcon=gcon.astype(np.float32), dx=dx
    )
    print(f"Geometry & Metric Saved. gcov shape: {gcov.shape}")

def process_dump(dump_id):
    try:
        pp.rblock_new(dump_id)
        pp.rpar_new(dump_id)
        pp.rgdump_direct()
        pp.rdump_griddata(DATA_DIR, dump_id)
        
        data = np.stack([
            pp.rho[0], pp.ug[0], 
            pp.uu[1][0], pp.uu[2][0], pp.uu[3][0],
            pp.B[1][0], pp.B[2][0], pp.B[3][0]
        ], axis=0)
        np.save(os.path.join(OUTPUT_DIR, f"dump_{dump_id}.npy"), data.astype(np.float32))
        return True
    except Exception as e:
        print(f"Error {dump_id}: {e}")
        return False

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    try: os.chdir(DATA_DIR)
    except: pass

    if rank == 0:
        save_static_grid(START_DUMP)
        all_dumps = list(range(START_DUMP, END_DUMP, STRIDE))
        chunks = np.array_split(all_dumps, size)
    else:
        chunks = None

    my_dumps = comm.scatter(chunks, root=0)
    for d in my_dumps: process_dump(d)
    
    comm.Barrier()
    if rank == 0: print("Processing Done.")

if __name__ == "__main__": main()