module ocean_model
  implicit none
  integer :: nsteps
  real :: dt
contains

subroutine schism_init(iths, ntime)
  implicit none
  integer, intent(out) :: iths
  integer, intent(in) :: ntime
  ! Initialize SCHISM model time stepping
  iths = 0
  ntime = nsteps
  call read_grid()
  call init_tracers()
end subroutine schism_init

subroutine timestep(it, dt_local)
  implicit none
  integer, intent(in) :: it
  real, intent(in) :: dt_local
  ! Advance one time step
  call solve_momentum(dt_local)
  call solve_transport(dt_local)
  call update_elevation(it)
end subroutine timestep

function compute_cfl(u, dx, dt_val) result(cfl)
  implicit none
  real, intent(in) :: u, dx, dt_val
  real :: cfl
  cfl = abs(u) * dt_val / dx
end function compute_cfl

end module ocean_model
