#ifndef __LITTER_H__
#define __LITTER_H__
#endif

__global__ void LitterDrainage(double *zliqsl, double *ph, double *dzlit_mm, double *drainlitter, double thetals, double thetafc, double km, double bm, double dt, int size);


__global__ void LitterWaterBalance(double *zliqsl, double *ph,  double *dzlit_mm,  double *ppt, double *drainlitter, double *Esl, double thetals, double dt, int size);

void LitterStorageModel(TimeForcingClass * &timeforcings, OverlandFlowClass * &overland_dev,
                       SubsurfaceFlowClass * &subsurface_dev, LitterSnowClass * &litter_dev, int rank, int procsize, int3 globsize, int t, int num_steps);
                       
void GatherLitterFluxes(ProjectClass *project, VerticalSoilClass *vertsoils, LitterSnowClass *litter_host, 
                        LitterSnowClass *litter_dev, SubsurfaceFlowClass *subsurface_dev, int rank, int procsize, 
                        int3 globsize, int3 domsize, int2 topolsize, int2 topolindex, MPI_Comm *cartComm);