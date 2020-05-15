class tempSphere {
public:
	vec3 center;
	vec3 veloctiy;
	vec3 color;
	float radius;
	tempSphere() {
		center = vec3(-5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5 - - 5))), -5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5 - - 5))), -5 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(5 - - 5))));
		veloctiy = vec3(-0.1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.1 - -0.1))), -0.1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.1 - - 0.1))), -0.1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.1 - - 0.1))));
		color = vec3(static_cast <float> (rand()) / static_cast <float> (RAND_MAX), static_cast <float> (rand()) / static_cast <float> (RAND_MAX), static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
		radius = 0.1;
	}
};