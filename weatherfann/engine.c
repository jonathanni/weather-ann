/*
 * =====================================================================================
 *
 *       Filename:  engine.c
 *
 *    Description:  Neural Network Engine for Weather Model
 *
 *        Version:  1.0
 *        Created:  03/01/2014 09:40:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Jonathan Ni 
 *   Organization:  Productive Productions
 *
 * =====================================================================================
 */

#include "include/fannarray.h"
#include "include/engine.h"
#include "include/vars.h"
#include "include/bool.h"
#include "include/logger.h"
#include "include/error.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <netcdf.h>
#include <math.h>
#include <fann.h>

// NetCDF ncids for files

int hgt_ncid = 0;
int pw_ncid = 0;
int rh_ncid = 0;
int sst_ncid = 0;
int tmp_ncid = 0;
int uwnd_ncid = 0;
int vwnd_ncid = 0;
int mask_ncid = 0;

// Time arrays

double * atime = 0;
double * astime = 0;

// Size of time arrays

size_t timelen = 0;
size_t stimelen = 0;

// Length of layer segments
const int lengths[6] = {7, 5, 5, 5, 5, 5};
const int offsets[6] = {0, \
                        (7) * GRID_SIZE, \
                        (7 + 5) * GRID_SIZE, \
                        (7 + 5 + 5) * GRID_SIZE, \
                        (7 + 5 + 5 + 5) * GRID_SIZE, \
                        (7 + 5 + 5 + 5 + 5) * GRID_SIZE};

// Min and Max for variable ranges (see what I did there? var_min -> varmin?)
static float var_min[7], var_max[7];

struct fann * network = 0;
struct fann_train_data * data = 0;

// Write NetCDF files
int forecastncid = 0;
int forecastvarids[11];

void openNetwork()
{
  FILE * file = fopen("../../network/network.net", "r");
  if(file == NULL)
    network = fann_create_sparse(CONNECTION_RATE, 4, SINGLE_SEGMENT * SAMPLE_SIZE, SINGLE_SEGMENT, SINGLE_SEGMENT, SINGLE_SEGMENT);
    //network = fann_create_shortcut(2, SINGLE_SEGMENT * SAMPLE_SIZE, SINGLE_SEGMENT);
  else
  {
    fclose(file);
    readNetwork();
  }

  setNetworkOptions();
}

void closeNetwork()
{
  fann_destroy(network);
}

void setNetworkOptions()
{
  fann_set_training_algorithm(network, FANN_TRAIN_QUICKPROP);
  fann_set_learning_rate(network, LEARNING_RATE);
  fann_set_activation_function_hidden(network, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(network, FANN_SIGMOID_SYMMETRIC);
}

void readNetwork()
{
  network = fann_create_from_file("../../network/network.net");
}

void writeNetwork()
{
  fann_save(network, "../../network/network.net");
}

void openData(char * year)
{
  char buf[128];
  
  printf("%s ", "Heights...");
  sprintf(buf, "%s%s%s", "../../data/HGT/hgt.", year, ".nc");
  e_netcdf(nc_open(buf, 0, &hgt_ncid));
  printf("%s\n", "OK");

  printf("%s ", "Precipitable Water...");
  sprintf(buf, "%s%s%s", "../../data/PW/pr_wtr.eatm.", year, ".nc");
  e_netcdf(nc_open(buf, 0, &pw_ncid)); 
  printf("%s\n", "OK");

  printf("%s ", "Relative Humidity...");
  sprintf(buf, "%s%s%s", "../../data/RH/rhum.", year, ".nc");
  e_netcdf(nc_open(buf, 0, &rh_ncid)); 
  printf("%s\n", "OK");
  
  printf("%s ", "Sea Surface Temperatures...");
  sprintf(buf, "%s%s%s", "../../data/SST/sst.", strtol(year, NULL, 10) < 1990 ? "1" : "2" , ".nc");
  e_netcdf(nc_open(buf, 0, &sst_ncid)); 
  printf("%s\n", "OK");
  
  printf("%s ", "Air Temperatures...");
  sprintf(buf, "%s%s%s", "../../data/TMP/air.", year, ".nc");
  e_netcdf(nc_open(buf, 0, &tmp_ncid)); 
  printf("%s\n", "OK");

  printf("%s ", "U Component Wind...");
  sprintf(buf, "%s%s%s", "../../data/UWND/uwnd.", year, ".nc");
  e_netcdf(nc_open(buf, 0, &uwnd_ncid));
  printf("%s\n", "OK");

  printf("%s ", "V Component Wind...");
  sprintf(buf, "%s%s%s", "../../data/VWND/vwnd.", year, ".nc");
  e_netcdf(nc_open(buf, 0, &vwnd_ncid));
  printf("%s\n", "OK");

  printf("%s ", "Land Sea Mask...");
  strcpy(buf, "../../data/SST/lsmask.1.nc");
  e_netcdf(nc_open(buf, 0, &mask_ncid));
  printf("%s\n", "OK");

  sprintf(buf, "%s%s\n", "Opened all NetCDF files for the year ", year);
  loginfo(buf);
}

void readData(fann_type * array, int timeind, int ssttimeind, int sample_length)
{

  int hgt_varid = 0;
  int pw_varid = 0;
  int rh_varid = 0;
  int sst_varid = 0;
  int tmp_varid = 0;
  int uwnd_varid = 0;
  int vwnd_varid = 0;

  e_netcdf(nc_inq_varid(hgt_ncid, "hgt", &hgt_varid));
  e_netcdf(nc_inq_varid(pw_ncid, "pr_wtr", &pw_varid));
  e_netcdf(nc_inq_varid(rh_ncid, "rhum", &rh_varid));
  e_netcdf(nc_inq_varid(sst_ncid, "sst", &sst_varid));
  e_netcdf(nc_inq_varid(tmp_ncid, "air", &tmp_varid));
  e_netcdf(nc_inq_varid(uwnd_ncid, "uwnd", &uwnd_varid));
  e_netcdf(nc_inq_varid(vwnd_ncid, "vwnd", &vwnd_varid));

  int ncids[5] = {hgt_ncid, rh_ncid, tmp_ncid, uwnd_ncid, vwnd_ncid};
  int varids[5] = {hgt_varid, rh_varid, tmp_varid, uwnd_varid, vwnd_varid};
  int sectors[5] = {HEGT, RELH, TEMP, UWND, VWND};

  int i = 0;
  for(i = 0; i < 5; i++)
    unpackLevels(timeind, ncids[i], varids[i], sectors[i], array, sample_length);
  unpackSingleLevel(timeind, pw_ncid, pw_varid, PREW, 0, array, false, sample_length);
  unpackSingleLevel(ssttimeind, sst_ncid, sst_varid, SSTP, 0, array, true, sample_length); 
}

void closeData()
{
  nc_close(hgt_ncid);
  nc_close(pw_ncid);
  nc_close(rh_ncid);
  nc_close(sst_ncid);
  nc_close(tmp_ncid);
  nc_close(uwnd_ncid);
  nc_close(vwnd_ncid);
  nc_close(mask_ncid);

  loginfo("Closed all NetCDF files\n");
}

void unpackSingleLevel(int timeind, int ncid, int varid, int sector, int level, \
                       fann_type * data, bool is_converted, int sample_length)
{
  // If the NetCDF file was converted using OpenGRaDS then it is of dimension 145x73, not 144x73
  // but thankfully the NetCDF nc_get_vara_type function supports selecting blocks of data more
  // gracefully. The fact that it is double * instead of short * is also noted. Also, all metadata
  // is lost for converted files so that is manually inputted (scale factor and offset).

  // Y + 1 goes south from Y, X - 1 goes west from X
//  size_t lstart[3] = {timeind, Y + 1, X - 1};
//  size_t lcount[3] = {1, HEIGHT, WIDTH};

  size_t lstart[3] = {timeind, 0, 0};
  size_t lcount[3] = {1, G_HEIGHT, G_WIDTH};

//  size_t istart[2] = {Y + 1, X - 1};
//  size_t icount[2] = {HEIGHT, WIDTH}; 
 
  size_t istart[2] = {0, 0};
  size_t icount[2] = {G_HEIGHT, G_WIDTH}; 

  float scale_factor = 0;
  float add_offset = 0;

  float bounds[2];

  if(is_converted)
  {
    double * buf = (double *) calloc(GRID_SIZE, sizeof(double));
    double * mask = (double *) calloc(GRID_SIZE, sizeof(double));

    scale_factor = 0.01f;
    add_offset = 0;

    bounds[0] = -5.0f;
    bounds[1] = 40.0f;

    var_min[sector] = bounds[0];
    var_max[sector] = bounds[1];

    int mask_varid = 0;
    e_netcdf(nc_inq_varid(mask_ncid, "mask", &mask_varid));
    
    e_netcdf(nc_get_vara_double(mask_ncid, mask_varid, istart, icount, mask));

    int i = 0;
    for(i = 0; i < sample_length; i++)
    {
      e_netcdf(nc_get_vara_double(ncid, varid, lstart, lcount, buf));
      unpackDoubleInto(buf, mask, data + i * SINGLE_SEGMENT + offsets[level], sector, \
                       scale_factor, add_offset, bounds[0], bounds[1]);
      lstart[0]--;
    }

    safeFree(buf);
    safeFree(mask);
  } else
  {
    short * buf = (short *) calloc(GRID_SIZE, sizeof(short));

    e_netcdf(nc_get_att_float(ncid, varid, "scale_factor", &scale_factor));
    e_netcdf(nc_get_att_float(ncid, varid, "add_offset", &add_offset));

    e_netcdf(nc_get_att_float(ncid, varid, "unpacked_valid_range", bounds));

    var_min[sector] = bounds[0];
    var_max[sector] = bounds[1];

    int i = 0;
    for(i = 0; i < sample_length; i++)
    {
      e_netcdf(nc_get_vara_short(ncid, varid, lstart, lcount, buf));
      unpackShortInto(buf, data + i * SINGLE_SEGMENT + offsets[level], sector, scale_factor, add_offset, bounds[0], bounds[1]);
      lstart[0]--;
    }

    safeFree(buf);
  }
}

void unpackLevel(int timeind, int ncid, int varid, int sector, int levind, int level, fann_type * data, int sample_length)
{
//  size_t lstart[4] = {timeind, levind, Y + 1, X - 1};
//  size_t lcount[4] = {1, 1, HEIGHT, WIDTH};

  size_t lstart[4] = {timeind, levind, 0, 0};
  size_t lcount[4] = {1, 1, G_HEIGHT, G_WIDTH};

  short * buf = (short *) calloc(GRID_SIZE, sizeof(short));
  float scale_factor = 0;
  float add_offset = 0;
  float bounds[2];

  e_netcdf(nc_get_att_float(ncid, varid, "scale_factor", &scale_factor));
  e_netcdf(nc_get_att_float(ncid, varid, "add_offset", &add_offset));

  e_netcdf(nc_get_att_float(ncid, varid, "unpacked_valid_range", bounds));

  var_min[sector] = bounds[0];
  var_max[sector] = bounds[1];

  int i = 0;
  for(i = 0; i < sample_length; i++)
  {
    e_netcdf(nc_get_vara_short(ncid, varid, lstart, lcount, buf));
    unpackShortInto(buf, data + i * SINGLE_SEGMENT + offsets[level], sector, scale_factor, add_offset, bounds[0], bounds[1]);
    lstart[0]--;
  }

  safeFree(buf);
}

void unpackLevels(int timeind, int ncid, int varid, int sector, fann_type * data, int sample_length)
{
  unpackLevel(timeind, ncid, varid, sector, 0, 0, data, sample_length);
  unpackLevel(timeind, ncid, varid, sector, 1, 1, data, sample_length);
  unpackLevel(timeind, ncid, varid, sector, 2, 2, data, sample_length);
  unpackLevel(timeind, ncid, varid, sector, 3, 3, data, sample_length);
  unpackLevel(timeind, ncid, varid, sector, 5, 4, data, sample_length);
  unpackLevel(timeind, ncid, varid, sector, 7, 5, data, sample_length);
}

void unpackShortInto(short * buf, fann_type * node, int sector, float scale_factor, float add_offset, \
		     float min, float max)
{
  int i = 0;
  for(i = 0; i < GRID_SIZE; i++)
    node[sector * GRID_SIZE + i] = ((((float) buf[i]) * scale_factor + add_offset) - min) / (max - min) * 2 - 1;
}

void unpackDoubleInto(double * buf, double * mask, fann_type * node, int sector, float scale_factor, float add_offset, \
		      float min, float max)
{
  int i = 0;
  for(i = 0; i < GRID_SIZE; i++)
    node[sector * GRID_SIZE + i] = (((((float) buf[i]) * scale_factor + add_offset) * mask[i]) - min) / (max - min) * 2 - 1;
}

char * getYear()
{
  int year = 0;
  loginfo("Year (1982-2013, anything else to exit)? ");
  if(scanf("%d", &year) != 1)
    exit(EXIT_FAILURE);

  if(year < 1982 || year > 2013)
    exit(EXIT_FAILURE);

  char * buf_year = (char *) calloc(5, sizeof(char));
  snprintf(buf_year, 5, "%d", year);

  return buf_year;
}

void openTimes()
{
  // Reused
  int time_varid = 0;
  int dimid[1];
  
  e_netcdf(nc_inq_varid(hgt_ncid, "time", &time_varid));
  e_netcdf(nc_inq_vardimid(hgt_ncid, time_varid, dimid));
  e_netcdf(nc_inq_dimlen(hgt_ncid, dimid[0], &timelen));
  
  atime = (double *) calloc(timelen, sizeof(double));
  

  e_netcdf(nc_inq_varid(sst_ncid, "time", &time_varid));
  e_netcdf(nc_inq_vardimid(sst_ncid, time_varid, dimid));
  e_netcdf(nc_inq_dimlen(sst_ncid, dimid[0], &stimelen));

  astime = (double *) calloc(stimelen, sizeof(double));
}

void closeTimes()
{
  safeFree(atime);
  safeFree(astime);
}

void openNetCDF(char * year, int index, int toffset)
{
  int i = 0;
  char buf[256];
  int datadimids[4], leveldimids[1], londimids[1], latdimids[1], timedimids[1];
  char * datanames[] = {"air_temperature", "heights", "rel_humidity", "u_wind", "v_wind", "sst", "pr_water"};

  snprintf(buf, 255, "../../forecast/forecast.%.4s.%d.nc", year, index);

  e_netcdf(nc_create(buf, NC_CLOBBER, &forecastncid));
  loginfo("Created NetCDF file...\n");

  e_netcdf(nc_def_dim(forecastncid, "time", 64, &datadimids[0]));
  e_netcdf(nc_def_dim(forecastncid, "level", 6, &datadimids[1]));
  e_netcdf(nc_def_dim(forecastncid, "lat", 3, &datadimids[2]));
  e_netcdf(nc_def_dim(forecastncid, "lon", 3, &datadimids[3]));

  loginfo("Created dimensions...\n");

  londimids[0]   = datadimids[3];
  latdimids[0]   = datadimids[2];
  leveldimids[0] = datadimids[1];
  timedimids[0]  = datadimids[0];

  for(i = 0; i < 7; i++)
    e_netcdf(nc_def_var(forecastncid, datanames[i], NC_FLOAT, 4, datadimids, &forecastvarids[i]));

  e_netcdf(nc_def_var(forecastncid, "level", NC_FLOAT, 1, leveldimids, &forecastvarids[7]));
  e_netcdf(nc_def_var(forecastncid, "lon", NC_FLOAT, 1, londimids, &forecastvarids[8]));
  e_netcdf(nc_def_var(forecastncid, "lat", NC_FLOAT, 1, latdimids, &forecastvarids[9]));
  e_netcdf(nc_def_var(forecastncid, "time", NC_INT, 1, timedimids, &forecastvarids[10]));
  
  loginfo("Defined all vars...\n");

  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[7], "units", 8, "millibar"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[8], "units", 11, "degree_east"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[9], "units", 12, "degree_north"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[10], "units", 27, "hours since 1-1-1 00:00:0.0"));

  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[7], "long_name", 5, "Level"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[8], "long_name", 9, "Longitude"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[9], "long_name", 8, "Latitude"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[10], "long_name", 4, "Time"));

  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[7],  "axis", 1, "Z"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[8],  "axis", 1, "X"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[9],  "axis", 1, "Y"));
  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[10], "axis", 1, "T"));

  e_netcdf(nc_put_att_text(forecastncid, forecastvarids[7], "positive", 4, "down"));

  const char * units[] = {"degK", "m", "%", "m/s", "m/s", "degC", "kg/m^2"};
  for(i = 0; i < 7; i++)
    e_netcdf(nc_put_att_text(forecastncid, forecastvarids[i], "units", strlen(units[i]), units[i]));

  loginfo("Defined all attributes..\n");

  e_netcdf(nc_enddef(forecastncid));
  
  // Begin definition for dimensions
  float levels[6] = {1000.0f, 925.0f, 850.0f, 700.0f, 500.0f, 300.0f};
  float lon[3] = {75, 77.5, 80};
  float lat[3] = {40, 37.5, 35};
  int times[64] = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, \
                   78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, \
                   150, 156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, \
                   222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, \
                   294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384};

  for(i = 0; i < 64; i++)
    times[i] += toffset;

  nc_put_var_float(forecastncid, forecastvarids[7], levels);
  nc_put_var_float(forecastncid, forecastvarids[8], lon);
  nc_put_var_float(forecastncid, forecastvarids[9], lat);
  nc_put_var_int(forecastncid, forecastvarids[10], times);
}

void closeNetCDF()
{
  e_netcdf(nc_close(forecastncid));
  loginfo("Closed NetCDF file...\n");
}

void writeNetCDF(fann_type * data, int timelapse)
{
  int i = 0, j = 0;

//  const size_t count[] = {1, 1, HEIGHT, WIDTH};
  const size_t count[] = {1, 1, G_HEIGHT, G_WIDTH};

  int k = 0, l = 0;
  for(i = 0; i < 6; i++)
  {
    const size_t start[] = {timelapse / 6 - 1, i, 0, 0};
    for(j = 0; j < lengths[i]; j++)
    {
      float scaled[GRID_SIZE];
//      for(k = 0; k < HEIGHT; k++)
//	for(l = 0; l < WIDTH; l++)
      for(k = 0; k < G_HEIGHT; k++)
	for(l = 0; l < G_WIDTH; l++)
/*	  scaled[k * WIDTH + l] = (data[offsets[i] + j * GRID_SIZE + k * WIDTH + l] + 1) / 2 \
                                * (var_max[j] - var_min[j]) + var_min[j];*/
	  scaled[k * G_WIDTH + l] = (data[offsets[i] + j * GRID_SIZE + k * G_WIDTH + l] + 1) / 2 \
                                * (var_max[j] - var_min[j]) + var_min[j];
      e_netcdf(nc_put_vara_float(forecastncid, forecastvarids[j], start, count, scaled));
    }
  }

  loginfo("Put all vars...\n"); 
}

void safeFree(void * ptr)
{
  free(ptr);
  ptr = NULL;
}

void train(char * year)
{
  openNetwork();

  //fann_print_connections(network);
  //return;

  loginfo("Reading time information...\n");
  char * buf_year = (strcmp(year, "-1") == 0 ? getYear() : year);
  openData(buf_year);
  safeFree(buf_year);

  loginfo("Reading time files...\n");
  openTimes();

  int i = 0, j = 0;
  fann_type * input = 0, * output = 0;

  input = (fann_type *) calloc(SAMPLE_SIZE * SINGLE_SEGMENT * BATCH_SIZE, sizeof(fann_type));
  output = (fann_type *) calloc(SINGLE_SEGMENT * BATCH_SIZE, sizeof(fann_type));

  for(i = SAMPLE_SIZE - 1; i < SAMPLE_SIZE - 1 + BATCH_SIZE; i++)
  {
    while((size_t) j < stimelen - 1 && astime[j] <= atime[i])
      ++j;

    loginfo("Read data\n");
    readData(input + (i - SAMPLE_SIZE + 1) * SAMPLE_SIZE * SINGLE_SEGMENT, i, j, SAMPLE_SIZE);
    readData(output + (i - SAMPLE_SIZE + 1) * SINGLE_SEGMENT, i + 1, j, 1);
  }

  struct fann_train_data * train = read_from_array(input, output, BATCH_SIZE, SAMPLE_SIZE * SINGLE_SEGMENT, SINGLE_SEGMENT);

  /*
  int test = 0;
  for(test = 0; test < SINGLE_SEGMENT; test++)
    printf("%f ", train->output[0][test]);   
  */

  fann_train_on_data(network, train, MAX_EPOCHS, EPOCHS_BETWEEN_REPORTS, DESIRED_ERROR);
  //fann_cascadetrain_on_data(network, train, MAX_NEURONS, NEURONS_BETWEEN_REPORTS, DESIRED_ERROR);
  
  safeFree(train);
  
  writeNetwork();

  safeFree(input);
  safeFree(output);

  closeTimes();

  closeData();
  closeNetwork();
}

void forecast(char * year, int index)
{
  openNetwork();

  char * buf_year = (strcmp(year, "-1") == 0 ? getYear() : year);
  openData(buf_year);

  loginfo("Reading time files...\n");
  openTimes();

  if(index < SAMPLE_SIZE - 1 || (size_t) index >= timelen)
  {
    logerr("Invalid input with index\n");
    exit(EXIT_FAILURE);
  }

  int j = 0;
  // input needs to be freed, but output points to ann->output so it doesn't
  fann_type * input = 0, * output = 0;
  input = (fann_type *) calloc(SAMPLE_SIZE * SINGLE_SEGMENT, sizeof(fann_type));

  // Initialize NetCDF file
  openNetCDF(buf_year, index, atime[index]);

  while((size_t) j < stimelen - 1 && astime[j] <= atime[index])
    ++j;

  int timelapse = 6;

  loginfo("Read data\n");
  readData(input, index, j, SAMPLE_SIZE);

  // forecast
  loginfo("Forecasting...\n");
  output = fann_run(network, input);

  writeNetCDF(output, timelapse);
  loginfo("Write NetCDF data ");
  printf("%s yr %d index %d hr\n", buf_year, index, timelapse);

  for(timelapse = 12; timelapse <= 384; timelapse += 6)
  {
    // loopback
    memmove(input + SINGLE_SEGMENT, input, SINGLE_SEGMENT * (SAMPLE_SIZE - 1));
    memcpy(input, output, SINGLE_SEGMENT);

    // forecast
    loginfo("Forecasting...\n");
    output = fann_run(network, input);

    //int aa = 0;
    //for(aa = 0; aa < SINGLE_SEGMENT; aa++) printf("%.2f ", output[aa]);

    writeNetCDF(output, timelapse); 
    loginfo("Write NetCDF data\n");
    printf("%s yr %d index %d hr\n", year, index, timelapse);
  }

  safeFree(input);
  safeFree(buf_year);

  closeNetCDF();
  closeTimes();
  closeData();
  closeNetwork();
}
