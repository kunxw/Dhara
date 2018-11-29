/*
// Copyright (C) 2018, HydroComplexity Group
// All rights reserved.
//
// Distributed Hydrologicc and Regional Analysis (DHARA) Model
// DHARA model is made available as a restricted, non-exclusive, 
// non-transferable license for education and research purpose only, 
// and not for commercial use. See the LICENSE.txt for more details.
//
// Author: kunxuanwang@yahoo.com (Kunxuan Wang)
*/

#include "../include/main.h"
#include "../include/cusplib.h"
#include "../include/devconst.h"
#include "../include/global.h"

__device__ double maxcomplsm (double a, double b)
{
    return (a < b) ? b : a;
}

/**
 * @brief      { calculate drainage of water from litter to soil }
 *
 * @param      zliqsl       [m] current water storage in litter
 * @param      ph           [m] water depth of ponding in overland
 * @param      dzlit        [m] depth of litter layer
 * @param      thetals      [] porosity
 * @param      thetafc      [] field capacity
 * @param      km           paramters 
 * @param      bm           paramters
 * @param      drainlitter  [m] amount of water drained
 * @param[in]  size        The size
 */
__global__ void LitterDrainage(double *zliqsl, double *ph, double *dzlit, double *drainlitter, double thetals, double thetafc, double km, double bm, double dt, int size) 
{
    // iv. Computation of Drainage
    // Clitter = zliqsl;
    // volliqli = zliqsl / dzlit_mm;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
    {
        double all_water = zliqsl[tid] + ph[tid]; // [m] consider ponded water
        double litter_water = all_water;
        if ((all_water / dzlit[tid]) > thetals ) { // check if over saturated 
            litter_water = dzlit[tid] * thetals;
        }
        if (all_water < 0) { // check for negative in ph (error in other modules of code)
            litter_water = 0;
        }
        
        // get drainage
        if ((litter_water / dzlit[tid]) < thetafc) {
            drainlitter[tid] = 0;
        } else {
            drainlitter[tid] = (km*exp(bm*(litter_water*1000.))*dt)/1000.;    // [m] Water that drains into soil from litter, equation is for mm, convert litter_water to mm first, then back to m           
        }
        
        // adjust ponding
        if(all_water > litter_water) {
            ph[tid] = all_water - litter_water;
        } else {
            ph[tid] = 0;
        }
        
        // Update threads if vector is long . . .
        tid += blockDim.x * gridDim.x;
    }   
}

/**
 * @brief      { rebalance water in litter and ponding }
 *
 * @param      zliqsl       [m] current water storage in litter
 * @param      ph           [m] water depth of ponding in overland
 * @param      dzlit        [m] depth of litter layer
 * @param      thetals      [] porosity
 * @param      ppt          [m] precip reaching the ground
 * @param      Esl          [mm/s] evaporation
 * @param      thetafc      [] field capacity
 * @param      drainlitter  [m] amount of water drained
 * @param[in]  size        The size
 */
__global__ void LitterWaterBalance(double *zliqsl, double *ph,  double *dzlit,  double *ppt, double *drainlitter, double *Esl, double thetals, double dt, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size)
    {
        double zliqsl_new = zliqsl[tid] + (ppt[tid] - drainlitter[tid] - Esl[tid] * sec_p_mm2dt_p_m * dt);  // [m liq]    
        if (zliqsl_new < 0) {   // check if negative water content
            drainlitter[tid] = maxcomplsm((drainlitter[tid] - (0 - zliqsl_new)), 0.0); // [m liq]
            zliqsl_new = 0;   // [m liq]
        }
        if ((zliqsl_new / dzlit[tid]) > thetals) {   // check if over saturated
            ph[tid] = ph[tid]+(zliqsl_new - dzlit[tid] * thetals);   //add extras to ponding
            zliqsl_new = dzlit[tid] * thetals;   // [m liq]
        }
        
        // update litter water storage
        zliqsl[tid] = zliqsl_new;
        ph[tid] = ph[tid] + drainlitter[tid];
        
        // Update threads if vector is long . . .
        tid += blockDim.x * gridDim.x;
    }
}

void LitterStorageModel(TimeForcingClass * &timeforcings, OverlandFlowClass * &overland_dev,
                       SubsurfaceFlowClass * &subsurface_dev, LitterSnowClass * &litter_dev, 
                       int rank, int procsize, int3 globsize, int t, int num_steps) 
{
    int sizexy  = globsize.x * globsize.y;
    
    /* 
    // overland_dev->ph       // Ponding height [L] used in subsurface
    overland_dev->waterdepth // ponding height used in overland
    subsurface_dev->ppt_ground 
    subsurface_dev->E_soil
    overland_dev->qcapa    // The flux capacity of soil [L/T]
    
    // calculate litter ET in Canopy model, pass to GPU like soilE
    Esl_liq = LEsl / (Lv_kg) * (1000/rho_liq);   //[mm liq / s] Evaporation From Sow-Litter Pack 
    Esl = Esl_liq; 
    */
    
    // get drainage amount and resulting ponding 
    LitterDrainage<<<TSZ,BSZ>>>(litter_dev->zliqsl, overland_dev->waterdepth, litter_dev->dzlit, 
                                litter_dev->drainlitter, litter_dev->thetals, litter_dev->thetafc, 
                                litter_dev->km, litter_dev->bm, litter_dev->dt, sizexy);    
    cudaCheckError("LitterDrainage");
    
    // vi. Compute water balance
    LitterWaterBalance<<<TSZ,BSZ>>>(litter_dev->zliqsl, overland_dev->waterdepth, 
                                    litter_dev->dzlit, subsurface_dev->ppt_ground, litter_dev->drainlitter, 
                                    litter_dev->Esl, litter_dev->thetals, litter_dev->dt, sizexy);
    cudaCheckError("LitterWaterBalance");

    SafeCudaCall( cudaMemcpy(overland_dev->ph, overland_dev->waterdepth, sizexy*sizeof(double),
            cudaMemcpyDeviceToDevice) );
}