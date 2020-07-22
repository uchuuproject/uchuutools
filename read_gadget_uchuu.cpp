#include<iostream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<cassert>

using namespace std;

#define IO_CACHE_SIZE (2097152)


class Particle{

 public:

  double m;
  double r[3];
  double v[3];

  long long int id;

  friend ostream & operator << (ostream &s, const Particle &p);


};


inline ostream & operator << (ostream &s, const Particle &p){

  s << p.m << " " 
    << p.r[0] << " " << p.r[1] << " " <<  p.r[2] << " " 
    << p.v[0] << " " << p.v[1] << " " <<  p.v[2] << " " 
    << " " << p.id;
  return s;

}



const double Gadget_UnitLength_in_Mpc = 1.0;            // 1Mpc
const double Gadget_UnitMass_in_Msun = 1.0e10;          // 1e10 Msun
const double Gadget_UnitVelocity_in_cm_per_s = 1e5;     //  1 km/sec
#define LONGIDS

typedef struct GadgetHeader{

  int      Npart[6];
  double   Massarr[6];
  double   Time;
  double   Redshift;
  int      FlagSfr;
  int      FlagFeedback;
  unsigned int Nall[6];
  int      FlagCooling;
  int      NumFiles;
  double   BoxSize;
  double   Omega0;
  double   OmegaLambda;
  double   HubbleParam;
  int      Flag_StellarAge;
  int      Flag_Metals;
  unsigned int NallHW[6];
  int      flag_entr_ics;
  char     unused[256- 6*4- 6*8- 2*8- 2*4- 6*4- 2*4 - 4*8 - 9*4];
  
}GadgetHeader, *pGadgetHeader;


inline int readGadget2( Particle *p, const char *filename){

  int nmemory = IO_CACHE_SIZE;
  GadgetHeader gadget_header;

  FILE *fin = fopen( filename, "rb");
  int blksize = 0;
  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "Reading Gadget file blksize= %d\n", blksize);
  fread( &gadget_header, sizeof(gadget_header), 1, fin);
  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "End Reading Gadget file blksize= %d\n", blksize);

  int npart = gadget_header.Npart[1];
  for( int i=0; i<npart; i++){
    p[i].m = gadget_header.Massarr[1];
  }

  float *cache = new float[nmemory*3];
  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "Reading Gadget file blksize= %d\n", blksize);
  for( int i=0; i<npart; i+=nmemory){
    int nread = nmemory;
    if( (i+nmemory) > npart)  nread = npart - i;
    fread( cache, sizeof(float), 3*nread, fin);
    for( int j=0; j<nread; j++){
      p[i+j].r[0] = cache[3*j+0];
      p[i+j].r[1] = cache[3*j+1];
      p[i+j].r[2] = cache[3*j+2];
    }
  }
  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "End Reading Gadget file blksize= %d\n", blksize);

  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "Reading Gadget file blksize= %d\n", blksize);
  for( int i=0; i<npart; i+=nmemory){
    int nread = nmemory;
    if( (i+nmemory) > npart)  nread = npart - i;
    fread( cache, sizeof(float), 3*nread, fin);
    for( int j=0; j<nread; j++){
      p[i+j].v[0] = cache[3*j+0];
      p[i+j].v[1] = cache[3*j+1];
      p[i+j].v[2] = cache[3*j+2];
    }
  }
  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "End Reading Gadget file blksize= %d\n", blksize);

  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "Reading Gadget file blksize= %d\n", blksize);

#ifndef LONGIDS
  int *icache = new int[nmemory];
  for( int i=0; i<npart; i+=nmemory){
    int nread = nmemory;
    if( (i+nmemory) > npart)  nread = npart - i;
    fread( icache, sizeof(int), nread, fin);
    for( int j=0; j<nread; j++){
      p[i+j].id = (long long int)icache[j];
    }
  }
#else
  long long int *icache = new long long int[nmemory];
  for( int i=0; i<npart; i+=nmemory){
    int nread = nmemory;
    if( (i+nmemory) > npart)  nread = npart - i;
    fread( icache, sizeof(long long int), nread, fin);
    for( int j=0; j<nread; j++){
      p[i+j].id = icache[j];
    }
  }
#endif
  fread( &blksize, 4, 1, fin);
  fprintf( stderr, "End Reading Gadget file blksize= %d\n", blksize);
  delete [] icache;

  fclose(fin);
  delete [] cache;


  return npart;

}



int main(int argc, char **argv){

  char *ic_filename = argv[1];
  int nmax = atoi( argv[2]);

  Particle *p = new Particle[nmax];
  int n = readGadget2( p, ic_filename);
  cerr << n << "\t" << nmax << endl;
  assert( n < nmax);

  for( int i=0; i<n; i++){
    cout << p[i] << endl;
  }

  delete [] p;

}
