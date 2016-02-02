#*******************************************************
#* Copyright (c) 2015 by Artelys                       *
#* All Rights Reserved                                 *
#*******************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Knitro example driver using reverse communications mode.
#
#  This executable invokes Knitro to solve a simple nonlinear
#  optimization test problem.  The purpose is to illustrate how to
#  invoke Knitro using the Python language API.
#
#  Before running, make sure ../../lib is in the load path.
#  To run:
#    python exampleQCQP
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from knitro import *


 ## Solve a small QCQP (quadratically constrained quadratic programming)
 #  test problem.
 #
 #  min   1000 - x0^2 - 2 x1^2 - x2^2 - x0 x1 - x0 x2
 #  s.t.  8 x0 + 14 x1 + 7 x2 - 56 = 0
 #        x0^2 + x1^2 + x2^2 - 25 >= 0
 #        x0 >= 0, x1 >= 0, x2 >= 0
 #
 #  The start point (2, 2, 2) converges to the minimum at (0, 0, 8),
 #  with final objective = 936.0.  From a different start point,
 #  Knitro may converge to an alternate local solution at (7, 0, 0),
 #  with objective = 951.0.
 ##

#----------------------------------------------------------------
#   METHOD evaluateFC
#----------------------------------------------------------------
 ## Compute the function and constraint values at x.
 #
 #  For more information about the arguments, refer to the Knitro
 #  manual, especially the section on the Callable Library.
 ##
def evaluateFC (x, c):

    #---- OBJECTIVE FUNCTION.
    dObj = 1.0e3 - x[0]*x[0] - 2.0e0*x[1]*x[1] - x[2]*x[2] \
                 - x[0]*x[1] - x[0]*x[2]
    
    #---- LINEAR EQUALITY CONSTRAINT.
    c[0] = 8.0e0*x[0] + 14.0e0*x[1] + 7.0e0*x[2] - 56.0e0
    
    #---- QUADRATIC INEQUALITY CONSTRAINT.
    c[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - 25.0e0
    
    return dObj


#----------------------------------------------------------------
#   METHOD evaluateGA
#----------------------------------------------------------------
 ## Compute the function and constraint first deriviatives at x.
 #
 #  For more information about the arguments, refer to the Knitro
 #  manual, especially the section on the Callable Library.
 ##
def evaluateGA (x, objGrad, jac):

    #---- GRADIENT OF THE OBJECTIVE FUNCTION.
    objGrad[0] = -2.0e0*x[0] - x[1] - x[2]
    objGrad[1] = -4.0e0*x[1] - x[0]
    objGrad[2] = -2.0e0*x[2] - x[0]
    
    #---- GRADIENT OF THE FIRST CONSTRAINT, c[0].
    jac[0] =  8.0e0
    jac[1] = 14.0e0
    jac[2] =  7.0e0
        
    #---- GRADIENT OF THE SECOND CONSTRAINT, c[1].
    jac[3] = 2.0e0*x[0]
    jac[4] = 2.0e0*x[1]
    jac[5] = 2.0e0*x[2]


#----------------------------------------------------------------
#   METHOD evaluateH
#----------------------------------------------------------------
 ## Compute the Hessian of the Lagrangian at x and lambda.
 #
 #  For more information about the arguments, refer to the Knitro
 #  manual, especially the section on the Callable Library.
 ##
def evaluateH (x, lambda_, sigma, hess):
    hess[0] = -2.0e0*sigma + 2.0e0*lambda_[1]
    hess[1] = -1.0e0*sigma
    hess[2] = -1.0e0*sigma
    hess[3] = -4.0e0*sigma + 2.0e0*lambda_[1]
    hess[4] = -2.0e0*sigma + 2.0e0*lambda_[1]


#----------------------------------------------------------------
#   MAIN METHOD FOR TESTING
#----------------------------------------------------------------

#---- DEFINE THE OPTIMIZATION TEST PROBLEM.
#---- FOR MORE INFORMATION ABOUT THE PROBLEM DEFINITION, REFER
#---- TO THE KNITRO MANUAL, ESPECIALLY THE SECTION ON THE
#---- CALLABLE LIBRARY.
n = 3
objGoal = KTR_OBJGOAL_MINIMIZE
objType = KTR_OBJTYPE_QUADRATIC;
bndsLo = [ 0.0 ] * 3
bndsUp = [ KTR_INFBOUND ] * 3
m = 2
cType = [ KTR_CONTYPE_LINEAR, KTR_CONTYPE_QUADRATIC ]
cBndsLo = [ 0.0, 0.0 ]
cBndsUp = [ 0.0, KTR_INFBOUND ]
nnzJ = 6
jacIxConstr = [ 0, 0, 0, 1, 1, 1 ]
jacIxVar    = [ 0, 1, 2, 0, 1, 2 ]
nnzH = 5
hessRow = [ 0, 0, 0, 1, 2 ]
hessCol = [ 0, 1, 2, 1, 2 ]

xInit = [ 2.0 ] * 3


#---- SETUP AND RUN KNITRO TO SOLVE THE PROBLEM.

#---- CREATE A NEW KNITRO SOLVER INSTANCE.
kc = KTR_new()
if kc == None:
    raise RuntimeError ("Failed to find a Ziena license.") #TODO

#---- DEMONSTRATE HOW TO SET KNITRO PARAMETERS.
if KTR_set_char_param_by_name(kc, "outlev", "all"):
    raise RuntimeError ("Error setting parameter 'outlev'")
if KTR_set_int_param_by_name(kc, "hessopt", 1):
    raise RuntimeError ("Error setting parameter 'hessopt'")
if KTR_set_int_param_by_name(kc, "hessian_no_f", 1):
    raise RuntimeError ("Error setting parameter 'hessian_no_f'")
if KTR_set_double_param_by_name(kc, "feastol", 1.0E-10):
    raise RuntimeError ("Error setting parameter 'feastol'")

#---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
ret = KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                cType, cBndsLo, cBndsUp,
                                jacIxVar, jacIxConstr,
                                hessRow, hessCol,
                                xInit, None)
if ret:
	raise RuntimeError ("Error initializing the problem, "
                                + "Knitro status = "
                                + str(ret))

#---- ALLOCATE ARRAYS FOR REVERSE COMMUNICATIONS OPERATION.
x       = [0] * n
lambda_  = [0] * (m + n)
obj     = [0]
c       = [0] * m
objGrad = [0] * n
jac     = [0] * nnzJ
hess    = [0] * nnzH

#---- SOLVE THE PROBLEM.  IN REVERSE COMMUNICATIONS MODE, KNITRO
#---- RETURNS WHENEVER IT NEEDS MORE PROBLEM INFORMATION.  THE CALLING
#---- PROGRAM MUST INTERPRET KNITRO'S RETURN STATUS AND CONTINUE
#---- SUPPLYING PROBLEM INFORMATION UNTIL KNITRO IS COMPLETE.
nEvalStatus = 0
nKnStatus = 1
while nKnStatus > 0:
    nKnStatus = KTR_solve (kc, x, lambda_, nEvalStatus, obj, c,
                                      objGrad, jac, hess, [], None)
    if nKnStatus == KTR_RC_EVALFC:
        #---- KNITRO WANTS obj AND c EVALUATED AT THE POINT x.
        obj[0] = evaluateFC (x, c)
    elif nKnStatus == KTR_RC_EVALGA:
        #---- KNITRO WANTS objGrad AND jac EVALUATED AT THE POINT x.
        evaluateGA (x, objGrad, jac)
    elif nKnStatus == KTR_RC_EVALH:
        #---- KNITRO WANTS hess EVALUATED AT THE POINT x.
        evaluateH (x, lambda_, 1.0, hess)
    elif nKnStatus == KTR_RC_EVALH_NO_F:
        #---- KNITRO WANTS hess EVALUATED AT THE POINT x
        #---- WITHOUT OBJECTIVE COMPONENT.
        evaluateH (x, lambda_, 0.0, hess)

    #---- ASSUME THAT PROBLEM EVALUATION IS ALWAYS SUCCESSFUL.
    #---- IF A FUNCTION OR ITS DERIVATIVE COULD NOT BE EVALUATED
    #---- AT THE GIVEN (x, lambda), THEN SET nEvalStatus = 1 BEFORE
    #---- CALLING solve AGAIN.
    nEvalStatus = 0

#---- DISPLAY THE RESULTS.
print "Knitro finished, status %d: " % nKnStatus
if nKnStatus == KTR_RC_OPTIMAL:
    print "converged to optimality."
elif nKnStatus == KTR_RC_ITER_LIMIT:
    print "reached the maximum number of allowed iterations."
elif nKnStatus in [KTR_RC_NEAR_OPT, KTR_RC_FEAS_XTOL, KTR_RC_FEAS_FTOL, KTR_RC_FEAS_NO_IMPROVE]:
    print "could not improve upon the current iterate."
elif nKnStatus == KTR_RC_TIME_LIMIT:
    print "reached the maximum CPU time allowed."
else:
    print "failed."

#---- EXAMPLES OF OBTAINING SOLUTION INFORMATION.
print "  optimal value = %f" % obj[0]
print "  solution feasibility violation    = %f" % KTR_get_abs_feas_error(kc)
print "           KKT optimality violation = %f" % KTR_get_abs_opt_error(kc)
print "  number of function evaluations    = %d" % KTR_get_number_FC_evals(kc)

#---- BE CERTAIN THE NATIVE OBJECT INSTANCE IS DESTROYED.
KTR_free(kc)

#+++++++++++++++++++ End of source file +++++++++++++++++++++++++++++
