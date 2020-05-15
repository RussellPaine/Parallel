#ifndef SPHEREH
#define SPHEREH
struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	vec3 color;
};

class sphere {
public:

	__device__ sphere(vec3 cen, float r) : center(cen), radius(r) {};
	__device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec);

	__device__ vec3 get_color();
	__device__ void update_position(vec3 systemBounds);
	__device__ void update_color(int colorType, vec3 *centerOfMass, sphere **list, int NumOfParticles);
	__device__ void update_gravity();
	__device__ float lerp(float X, float A, float B, float C, float D);

	vec3 center;
	vec3 veloctiy;
	vec3 color;
	vec3 originalColor;
	float radius;
	__device__ sphere(vec3 cen, vec3 vel, vec3 co, float r) {
		center =  cen;
		veloctiy = vel;
		color = co;
		radius = r;
		originalColor = co;
	}
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) 
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.color = sphere::get_color();
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.color = sphere::get_color();
			return true;
		}
	}
	return false;
}

__device__ vec3 sphere::get_color()
{
	return color;
}

__device__ void sphere::update_color(int colorType, vec3 *centerOfMass, sphere **list, int NumOfParticles) {
	int count = 1;
	switch (colorType) {
		case 1:
			color = originalColor;
			return;
		case 2:
			color = vec3(lerp(sqrt(pow(veloctiy[0], 2) + pow(veloctiy[1], 2) +pow(veloctiy[2], 2)), 0, 0.12, 0, 1), 0, 0);
			return;
		case 3:
			for (int i = 0; i < NumOfParticles; i++) if (sqrt(pow(center[0] - list[i]->center[0], 2) + pow(center[1] - list[i]->center[1], 2) +pow(center[2] - list[i]->center[2], 2)) < 2) count++;
			color = vec3(0,lerp(count, 0, 100, 0, 1), 0);
			return;
		case 4:
			float c = 1 - lerp(sqrt(pow(center[0] - centerOfMass->x(), 2) + pow(center[1]- centerOfMass->y(), 2) + pow(center[2] - centerOfMass->z(), 2)), 0, 10, 0, 1);
			color = vec3(0, 0, c);
			return;
	};
}

__device__ float sphere::lerp(float X, float A, float B, float C, float D) {
  return (X-A)/(B-A) * (D-C) + C;
}

__device__ void sphere::update_position(vec3 systemBounds) {
	center += veloctiy;

	if (center[0] > systemBounds[0] / 2) {
		veloctiy[0] = -veloctiy[0];
		center[0] = systemBounds[0] / 2;
	}
	if (center[0] < -systemBounds[0] / 2) {
		veloctiy[0] = -veloctiy[0];
		center[0] = -systemBounds[0] / 2;
	}

	if (center[1] > systemBounds[1] / 2) {
		veloctiy[1] = -veloctiy[1];
		center[1] = systemBounds[0] / 2;
	}
	if (center[1] < -systemBounds[1] / 2) {
		veloctiy[1] = -veloctiy[1];
		center[1] = -systemBounds[0] / 2;
	}

	if (center[2] > systemBounds[2] / 2) {
		veloctiy[2] = -veloctiy[2];
		center[2] = systemBounds[0] / 2;
	}
	if (center[2] < -systemBounds[2] / 2) {
		veloctiy[2] = -veloctiy[2];
		center[2] = -systemBounds[0] / 2;
	}


}

__device__ void sphere::update_gravity() {
	center[1] = center[1] - 1;
}

#endif
