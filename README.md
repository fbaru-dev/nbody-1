# nbody
n-body solver, with the aim of writing a fast MPI implementation.

For the n-body problem this is a little non-trivial since each timestep the position of every body is needed, so it is something of a communication-heavy problem. Cluster available for testing has only a Gigabit Ethernet interconnect, so efficient communication will be essential.



### Serial Implementation
Particle position, velocity, acceleration and mass are each stored in separate arrays rather than a somewhat neater struct approach, to make it easier for the auto-vectorizer and to use manual vector intrinsics.

#### Manual Vectorization
Compute accelerations for 4 particles at a time in inner loop of `ComputeAccel`, routine called `ComputeAccelAVX`. This is just over twice as fast as the autovectorized version.
