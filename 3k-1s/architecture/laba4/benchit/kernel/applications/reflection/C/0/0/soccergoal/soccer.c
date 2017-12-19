/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: soccer.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/reflection/C/0/0/soccergoal/soccer.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the possibility to hit the goal when the pole have a squared or round form
 *******************************************************************/

/*
 *  explanation:
 *    * we have 2 circles (in the simulation only halfcircles, because a shot
 *                         from a point outside of the field make no sense):
 *        - one is for the point the player stand with the ball
 *        - the next is for point the ball hits the pole
 *    * the player make a shot from each point of the outer circle to each point of the inner circle
 *    * now we count the balls which end up in the goal
 *        - one time for a goal with squared poles
 *        - one time for a goal with round poles
 */

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "interface.h"
#include "soccer.h"

void coordtrafo(point_t *, double , double , point_t *);
void coordtrafo_square(point_t *, double , double , point_t *, int *);
void getline(line_t *, point_t *, point_t *);
void getline_square(line_t *, point_t *, point_t *, int *);
void getmirror(line_t *, line_t *, line_t *, point_t *);
int intersec_withgoalline(line_t *, point_t *, point_t *);
int intersec_withroundpole(line_t *, point_t *, point_t *, point_t *, double);
int intersec_withsquarepole(line_t *, point_t *, point_t *, point_t *, double , int *);

void plotline(line_t *);


/* dimensions soccer goal:
 *   pole: 12*12cm
 *   goal: 7.32m between inner side of the poles
 *         -> 744cm is the distance between center of the poles
 */
double soccergoal(myinttype *probsize, myinttype *version){
  myinttype i, cur, hitgoal, hitpole, count=*probsize, side;
  
  double pfostenradius=6.0;
  double schussradius=100.0;
  double PI = 3.141592653589793;
  
  point_t pfosten[2];
  point_t pfosten1={0.0, 0.0};
  point_t pfosten2={744.0, 0.0};
  
  point_t schuss_start={0.0, 0.0};
  point_t pfostenkontakt={0.0, 0.0};
  
  line_t line, mirroraxes, mirror;
  
  /* alpha: angle outer circle (point of the player)
   * beta:  angle inner circle (pole)
   */
  double alpha, beta, goals;
  
  pfosten[0] = pfosten1;
  pfosten[1] = pfosten2;

  goals = 0.0;

  IDL(3, printf("\nversion: %i\n",*version));

  for(alpha=0.0; alpha<=PI; alpha+=(PI/count)){
    for(beta=0.0; beta<=PI; beta+=(PI/count)){
      IDL(3, printf("\nnext shot: alpha=%e, beta=%e, version=%i\n", alpha, beta, *version));

      hitgoal = 0;
      hitpole = 1;
      cur = 0;
      /* start point of the shot */
      coordtrafo(&schuss_start, schussradius, alpha, &pfosten[cur]);
      
      if(*version==0){
        /* contact point of the shot with the square pole */
        coordtrafo_square(&pfostenkontakt, pfostenradius, beta, &pfosten[cur], &side);
      } else {
        /* contact point of the shot with the round pole */
        coordtrafo(&pfostenkontakt, pfostenradius, beta, &pfosten[cur]);
      }

      /* the ball can only jump 5 times from one pole to the other */
      for(i=0; i<5; i++){
        cur = i - ((i>>1)<<1); 
        getline(&line, &pfostenkontakt, &schuss_start);
        IDL(3, printf("pcontx,pconty=%e, %e\n",pfostenkontakt.x,pfostenkontakt.y));
        
        /* transform the hit point, because the ball can not tunnel thru the pole
         * (you can shot from each point to each point, but sometimes the pole blockade the way)
         */
        if(i==0){
          line.sx = line.sx + 100*line.tx;
          line.sy = line.sy + 100*line.ty;
          line.tx = -line.tx;
          line.ty = -line.ty;
          if(*version==0){
            intersec_withsquarepole(&line, &pfosten[cur], &schuss_start, &pfostenkontakt, pfostenradius, &side);
          } else {
            intersec_withroundpole(&line, &pfosten[cur], &schuss_start, &pfostenkontakt, pfostenradius);
          }
          getline(&line, &pfostenkontakt, &schuss_start);
        }
        IDL(3, plotline(&line));

        if(*version==0){
          /* get the axis of reflection for square pole */
          getline_square(&mirroraxes, &pfosten[cur], &pfostenkontakt, &side);
        } else {
          /* get the axis of reflection for round pole */
          getline(&mirroraxes, &pfosten[cur], &pfostenkontakt);
        }
        IDL(3, plotline(&mirroraxes));

        /* calc the next way of the ball after hit the pole */
        getmirror(&mirror, &line, &mirroraxes, &pfostenkontakt);
        IDL(3, plotline(&mirror));

        /* test if the reflected ball ends up in the goal,
         *  -> yes: count the goal and try the next shot
         *  -> no:  test if the reflected ball hits the other pole */
        hitgoal = intersec_withgoalline(&mirror, &pfosten[0], &pfosten[1]);
        if(hitgoal){
          goals = goals + 1;
          break;
        } else {
          if(*version==0){
            hitpole = intersec_withsquarepole(&mirror, &pfosten[cur^1], &schuss_start, &pfostenkontakt, pfostenradius, &side);
          } else {
            hitpole = intersec_withroundpole(&mirror, &pfosten[cur^1], &schuss_start, &pfostenkontakt, pfostenradius);
          }
          if(hitpole==0){
            break;
          }
        }
        
      }
      
    }
  }
  
  return goals;
}

/* transform the coordinates of the parameter representation into cartesian coordinates */
/*                        OUT           IN             IN            IN  */
void coordtrafo(point_t *point, double radius, double arc, point_t *m){
  point->x = m->x + radius * cos(arc);
  point->y = m->y + radius * sin(arc);
}

/* transform the coordinates of the parameter representation into cartesian coordinates
 * and calc the intersection with the square
 */
/*                               OUT           IN             IN            IN      OUT    */
void coordtrafo_square(point_t *point, double radius, double arc, point_t *m, int *side){
  line_t line;
  double p, r;
  double PI = 3.141592653589793;
    
  if(arc<0.25*PI){
    /* right side of the pole */
    *side = 1;
    p = m->x + radius;
  } else if(arc<0.75*PI){
    /* front side of the pole */
    *side = 0;
    p = m->y + radius;
  } else {
    /* left side of the pole */
    *side = -1;
    p = m->x - radius;
  }
  
  line.sx = m->x;
  line.sy = m->y;
  line.tx = radius * cos(arc);
  line.ty = radius * sin(arc);
  
  if(*side==0){
    /* p = sy + r * ty */
    r = (p - line.sy) / line.ty;
  } else {
    /* p = sx + r * tx */
    r = (p - line.sx) / line.tx;
  }
  
  point->x = line.sx + r * line.tx;
  point->y = line.sy + r * line.ty;
}

/* calc the line between two points, with unit direction vector */
/*                    OUT            IN               IN       */
void getline(line_t *line, point_t *point1, point_t *point2){
  double fac;

  line->sx = point1->x;
  line->sy = point1->y;
  line->tx = point2->x - point1->x;
  line->ty = point2->y - point1->y;

  fac = sqrt(line->tx*line->tx + line->ty*line->ty);
  line->tx /= fac;
  line->ty /= fac;
}

/* calc the line between two points, with unit direction vector, its for the normal vectors
 * (respectively the line of the mirror axis) of the squared pole */
/*                           OUT            IN               IN           IN     */
void getline_square(line_t *line, point_t *point1, point_t *point2, int *side){
  if(*side==1){
    line->sx = point1->x;
    line->sy = point2->y;
    line->tx = 1;
    line->ty = 0;
  } else if(*side==0){
    line->sx = point2->x;
    line->sy = point1->y;
    line->tx = 0;
    line->ty = 1;
  } else if(*side==-1){
    line->sx = point1->x;
    line->sy = point2->y;
    line->tx = -1;
    line->ty = 0;
  }
}

/* calc the reflected line, with unit direction vector
 * (first tested a reflection with rotation matrix, but the result was not accurate enough */
/*                      OUT             IN            IN             IN     */
void getmirror(line_t *mirror, line_t *line, line_t *axis, point_t *intersec){
  double s1, s2, t1, t2, u1, u2, v1, v2;
  double ix, iy, mx, my;
  double b, fac;
  
  s1 = line->sx + line->tx;
  s2 = line->sy + line->ty;
  t1 = -axis->ty;
  t2 = axis->tx;
  
  u1 = axis->sx;
  u2 = axis->sy;
  v1 = axis->tx;
  v2 = axis->ty;
  
  IDL(3, printf("s1,s2,t1,t2=%e, %e, %e, %e;  u1,u2,v1,v2=%e, %e, %e, %e;  \n",s1,s2,t1,t2,u1,u2,v1,v2));

  /* calc for intersection of a new line (build from a point on the line and the normal vector of the axis) and axis */
  if(t1==0){
    b = (s1-u1) / v1;
  } else {
    b = (s2/v2 + (u1-s1)*t2/(t1*v2) - u2) / (1 - (t2*v1)/(t1*v2));
  }
  
  /* intersection point of the new line and the axis */
  ix = u1 + b*v1;
  iy = u2 + b*v2;
  
  /* make a point reflection */
  mx = s1 + 2.0*(ix - s1);
  my = s2 + 2.0*(iy - s2);
  
  /* calc the reflected line out of the reflected point and the intersection point of the axis and the old line */
  mirror->sx = intersec->x;
  mirror->sy = intersec->y;
  mirror->tx = mx - mirror->sx;
  mirror->ty = my - mirror->sy;

  fac = sqrt(mirror->tx*mirror->tx + mirror->ty*mirror->ty);
  mirror->tx /= fac;
  mirror->ty /= fac;
}

/* calc the reflected line, with unit direction vector */
/*                      OUT             IN            IN             IN     */
int intersec_withgoalline(line_t *line, point_t *p1, point_t *p2){
  double a, b;
  
  if(line->ty > 0.0){
    return 0;
  }

  if(line->ty != 0.0){
    a = -1.0 * line->sy / line->ty;
    b = line->sx + a * line->tx;
  } else {
    return 0;
  }
  
  if(b>=p1->x && b<=p2->x){
    return 1;
  }else{
    return 0;
  }
}

/* (s1-m1)^2 + (s2-m2)^2 - r^2 + 2x(t1(s1-m1) + t2(s2-m2)) + x^2(t1^2+t2^2)  =  0
 * solution: x_1/2 = (-2(t1(s1-m1) + t2(s2-m2)) +- sqrt(4(t1(s1-m1) + t2(s2-m2))^2 - 4(t1^2+t2^2)((s1-m1)^2 + (s2-m2)^2 - r^2))) / 2(t1^2+t2^2)
 */
/* calc the intersection point of a line with the round pole
 * return 0 if there is no intersection, and 1 for an intersec. */
/*                                  IN             IN                OUT             OUT              IN       */
int intersec_withroundpole(line_t *line, point_t *pfosten, point_t *point, point_t *intersec, double radius){
  double s1=0.0, s2=0.0, t1=0.0, t2=0.0, m1=0.0, m2=0.0, temp1=0.0, temp2=0.0, r=0.0;
  double x1=0.0, x2=0.0;
  myinttype ret=1;
  
  s1 = line->sx;
  s2 = line->sy;
  t1 = line->tx;
  t2 = line->ty;
  m1 = pfosten->x;
  m2 = pfosten->y;
  r = radius;

  temp1 = 4 * (t1*(s1-m1) + t2*(s2-m2)) * (t1*(s1-m1) + t2*(s2-m2)) - 4 * (t1*t1+t2*t2)*((s1-m1)*(s1-m1) + (s2-m2)*(s2-m2) - r*r);

  if(temp1>0.0){
    x1 = (-2*(t1*(s1-m1) + t2*(s2-m2)) + sqrt(temp1)) / (2*(t1*t1+t2*t2));
    x2 = (-2*(t1*(s1-m1) + t2*(s2-m2)) - sqrt(temp1)) / (2*(t1*t1+t2*t2));
  } else {
    ret = 0;
  }
  
  /* temorary x values of the "two" possible intersection points */
  temp1 = s1 + x1*t1;
  temp2 = s1 + x2*t1;
  
  /* get this intersection point, which is closer to the "shot" point */
  if(fabs(temp1-line->sx) < fabs(temp2-line->sx)){
    intersec->x = temp1;
    intersec->y = s2 + x1*t2;
  } else {
    intersec->x = temp2;
    intersec->y = s2 + x2*t2;
  }

  if((intersec->x < 0.0) != (line->sx < 0.0)) ret = 0;
  
  point->x = line->sx;
  point->y = line->sy;

  return ret;
}

/* calc the intersection point of a line with the squared pole
 * return 0 if there is no intersection, and 1 for an intersec. */
/*                                   IN             IN                OUT             OUT              IN           OUT    */
int intersec_withsquarepole(line_t *line, point_t *pfosten, point_t *point, point_t *intersec, double radius, int *side){
  myinttype ret[3], min, tside[3]={1, 0, -1};
  point_t p[3];
  double r1, r2, r3, abs[3];

  /* with right: p = pfosten.x + radius = sx + r * tx */
  r1 = (pfosten->x + radius - line->sx) / line->tx;
  p[0].x = line->sx + r1 * line->tx;
  p[0].y = line->sy + r1 * line->ty;
  ret[0] = (0.0<=p[0].y && p[0].y<=radius) ? 1 : 0;
  IDL(3, printf("p[0].x,p[0].y=%e, %e,  ret[0]=%i\n",p[0].x,p[0].y, ret[0]));

  /* with front: p = pfosten.y + radius = sy + r * ty */
  r2 = (pfosten->y + radius - line->sy) / line->ty;
  p[1].x = line->sx + r2 * line->tx;
  p[1].y = line->sy + r2 * line->ty;
  ret[1] = (pfosten->x-radius<=p[1].x && p[1].x<=pfosten->x+radius) ? 1 : 0;
  IDL(3, printf("p[1].x,p[1].y=%e, %e,  ret[1]=%i\n",p[1].x,p[1].y, ret[1]));
  
  /* with left: p = pfosten.x + radius = sx + r * tx */
  r3 = (pfosten->x - radius - line->sx) / line->tx;
  p[2].x = line->sx + r3 * line->tx;
  p[2].y = line->sy + r3 * line->ty;
  ret[2] = (0.0<=p[2].y && p[2].y<=radius) ? 1 : 0;
  IDL(3, printf("p[2].x,p[2].y=%e, %e,  ret[2]=%i\n",p[2].x,p[2].y, ret[2]));

  abs[0] = sqrt((p[0].x-line->sx)*(p[0].x-line->sx) + (p[0].y-line->sy)*(p[0].y-line->sy));
  abs[1] = sqrt((p[1].x-line->sx)*(p[1].x-line->sx) + (p[1].y-line->sy)*(p[1].y-line->sy));
  abs[2] = sqrt((p[2].x-line->sx)*(p[2].x-line->sx) + (p[2].y-line->sy)*(p[2].y-line->sy));
  abs[0] = (isnan(abs[0]) || ret[0]==0) ? DBL_MAX : abs[0];
  abs[1] = (isnan(abs[1]) || ret[1]==0) ? DBL_MAX : abs[1];
  abs[2] = (isnan(abs[2]) || ret[2]==0) ? DBL_MAX : abs[2];
  IDL(3, printf("abs1, abs2, abs3 = %e, %e, %e\n", abs[0], abs[1], abs[2]));

  min = (abs[0]<abs[1]) ? 0 : 1;
  min = (abs[2]<abs[min]) ? 2 : min;
  IDL(3, printf("min = %i\n", min));

  *intersec = p[min];
  *side = tside[min];
    
  if((intersec->x < 0.0) != (line->sx < 0.0)) ret[min] = 0;

  point->x = line->sx;
  point->y = line->sy;

  return ret[min];
}

/* print two points of a line with distance 200 (because of the unit direction vector) */
/*                     IN     */
void plotline(line_t *line){
  printf("%3.16e, %3.16e\n", line->sx, line->sy);
  printf("%3.16e, %3.16e\n", line->sx+200*line->tx, line->sy+200*line->ty);
}


