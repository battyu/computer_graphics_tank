#include "framework.h"

const int tessellationLevel = 20;
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;		// intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }
};
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};
class PiramisTexture : public Texture {
//---------------------------
public:
    PiramisTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 brown(0.8f, 0.9f, 0.1f, 1.0f);

        for (int i = 0; i < width * height; ++i) {
            image[i] = brown;
        }

        create(width, height, image, GL_NEAREST);
    }
};
struct RenderState {
//---------------------------
    mat4	           MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    vec3	           wEye;
};
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};
class PhongShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

    // fragment shader in GLSL
    const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La +
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use(); 		// make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        //setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");

        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};
class Geometry {
//---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};
class PolygonMesh : public Geometry {
//---------------------------

protected:
    struct VertexData {
        vec3 position, normal;
    };

    std::vector<vec3> faces;
    std::vector<vec3> vertices;

public:

    PolygonMesh() {
        this->create();
    }

    VertexData GenVertexData(vec3 csucs, vec3 tobbicsucs) {
        VertexData vtxData;
        vtxData.position = csucs;

        vec3 vertex1 = vertices[tobbicsucs.x];
        vec3 vertex2 = vertices[tobbicsucs.y];
        vec3 vertex3 = vertices[tobbicsucs.z];

        vec3 oldalvec1 = vertex2 - vertex1;
        vec3 oldalvec2 = vertex3 - vertex1;

        vtxData.normal = normalize(cross(oldalvec1,oldalvec2));

        return vtxData;
    }

    void create() {

        std::vector<VertexData> vtxData; // vertices on the CPU

        for (int i = 0; i < faces.size(); i++) {
            for (int j = 0; j < vertices.size(); j++) {
                vtxData.push_back(GenVertexData(vertices[j], faces[i]));


            }
        }

        glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));

    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < vertices.size(); i++) glDrawArrays(GL_TRIANGLES, i , i * vertices.size());
    }
};
class Piramis : public PolygonMesh{
public:
    Piramis(){
        vertices = {
                {0.0f, 0.0f, 0.0f},
                {1.0f, 0.0f, 0.0f},
                {1.0f, 0.0f, 1.0f},
                {0.0f, 0.0f, 1.0f},
                {0.5f, 1.6f, 0.5f}
        };
        faces = {
                {0, 1, 2},
                {0, 2, 3},
                {0, 3, 4},
                {3, 2, 4},
                {2, 1, 4},
                {1, 0, 4}
        };
        this->create();
    }
};
class Quad : public PolygonMesh{
public:
    Quad(){
        vertices = {{0.0f, 0.0f, 0.0f},
                    {1.0f, 0.0f, 0.0f},
                    {1.0f, 0.0f, 0.3f},
                    {0.0f, 0.0f, 0.3f}};
        faces = {{0,1,2},
                 {0,2,3}};
        this->create();
    }
};
struct Object {
//---------------------------
    Shader *   shader;
    Material * material;
    Geometry * geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
    float l;
public:
    Object(Shader * _shader, Material * _material, /*Texture * _texture,*/ Geometry * _geometry) :
            scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        material = _material;
        geometry = _geometry;
    }

    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        shader->Bind(state);
        geometry->Draw();
    }

    vec3 getLookat(){
        return translation;
    }

};
class TrackShoe {
    std::vector<Object *> quads;
    float seb;
    const float radius = 0.5;
public:
    TrackShoe(Shader * shader, Material * material, vec3 center, int side){
        seb = 0;
        float x;
        if(side == 0){
            x = center.x-0.7;
        }
        else if(side==1){
            x = center.x+0.7;
        }
        for(int i = 0; i<12; i++){
            Geometry * quad = new Quad();
            Object * talpObject = new Object(shader, material, quad);
            talpObject->translation = vec3(x, center.y+5, center.z+0.3*i);
            talpObject->scale = vec3(0.5, 0.5, 0.5);
            talpObject->rotationAxis = vec3(1, 0, 0);
            talpObject->rotationAngle = (float)M_PI / 180.0f  ;
            talpObject->l=i*(radius+float(M_PI)*radius/6);
            quads.push_back(talpObject);
        }
    }
    std::vector<Object *> getQuads(){return quads;}
    float getSeb(){return seb;}
    void setSeb(float v){seb=v;}

    void move(vec3 kozep, float dt, vec3 h, float alpha){
        for(int i = 0; i<quads.size(); i++){
            quads.at(i)->l = quads.at(i)->l + dt*seb;
            quads.at(i)->rotationAxis = vec3(0,1,0);
            quads.at(i)->rotationAngle = -alpha+float(M_PI)/2;
            if(quads.at(i)->l>=12*radius+2*float(M_PI)*radius){
                quads.at(i)->l = 0;
            }
            if(quads.at(i)->l<0-0.000001){
                quads.at(i)->l = 12*radius+2*float(M_PI)*radius-0.000001;
            }
            if(quads.at(i)->l<6*radius){
                quads.at(i)->translation = kozep + h*(3*radius - quads.at(i)->l);
                quads.at(i)->translation.y = kozep.y-radius;
            }
            if(quads.at(i)->l>=6*radius && quads.at(i)->l<6*radius+float(M_PI)*radius){
                quads.at(i)->translation = kozep - h*3*radius - h*radius*sin((quads.at(i)->l-6*radius)/radius);
                quads.at(i)->translation.y = kozep.y - radius*cos((quads.at(i)->l-6*radius)/radius);
            }
            if(quads.at(i)->l>=6*radius+float(M_PI)*radius && quads.at(i)->l<12*radius+float(M_PI)*radius){
                quads.at(i)->translation = kozep - h*(3*radius - (quads.at(i)->l)+6*radius+float(M_PI)*radius);
                quads.at(i)->translation.y = kozep.y+radius;
            }
            if(quads.at(i)->l>=12*radius+float(M_PI)*radius && quads.at(i)->l<12*radius+2*float(M_PI)*radius){
                float dl = quads.at(i)->l-12*radius-float(M_PI)*radius;
                quads.at(i)->translation = kozep + h*3*radius + h*radius*sin(dl/radius);
                quads.at(i)->translation.y = kozep.y + radius*cos(dl/radius);
            }
        }
    }
};
class Tank {
    vec3 h;
    vec3 v_center;
    vec3 center;
    TrackShoe * right;
    TrackShoe * left;
    float alpha;
    float w;
public:

    Tank(vec3 c, Shader* shader, Material* material){
        right = new TrackShoe(shader, material, c, 0); //0: jobboldali
        left = new TrackShoe(shader, material, c, 1); //1: baloldali
        center = c;
        alpha = 0.523599;
        h = vec3(cos(alpha), sin(alpha), 0);
        v_center = (h*(right->getSeb()+left->getSeb())/2);
    }
    TrackShoe * getRight(){
        return right;
    }
    TrackShoe * getLeft(){
        return left;
    }
    void setSebR(float v){
        right->setSeb(v);
    }
    void setSebL(float v){
        left->setSeb(v);
    }
    vec3 getCenter(){
        return center;
    }
    vec3 getH(){return h;}
    void Animate(float dt, float vj, float vb){
        if(vj!=0 || vb!=0){
            center = center + v_center*dt;
            w = (vj-vb)/1.4;
            alpha = alpha + w*dt;
            h = vec3(cos(alpha), 0, sin(alpha));
            v_center = h*(vj+vb)/2;

            vec3 jobbkozep = center + vec3(sin(alpha),0,-cos(alpha))*0.7;
            right->move(jobbkozep,  dt, h, alpha);
            vec3 balkozep = center + vec3(-sin(alpha),0,cos(alpha))*0.7;
            left->move(balkozep,  dt, h, alpha);
        }
    }
};
class Scene {
//---------------------------
    Camera camera; // 3D camera
    std::vector<Light> lights;
    std::vector<Object *> objects;
    std::vector<Object *> rightShoe;
    std::vector<Object *> leftShoe;
    Tank * tank;

public:
    Tank* getTank() {return tank;}
    void Build() {
        // Shaders
        Shader * phongShader = new PhongShader();

        // Materials
        Material * material0 = new Material;
        material0->kd = vec3(0.4f, 0.2f, 0.0f);
        material0->ks = vec3(4, 4, 4);
        material0->ka = vec3(0.1f, 0.1f, 0.1f);
        material0->shininess = 10000;

        Material * material1 = new Material;
        material1->kd = vec3(0.76f, 0.6f, 0.0f);
        material1->ks = vec3(4, 4, 4);
        material1->ka = vec3(0.1f, 0.1f, 0.1f);
        material1->shininess = 10000;

        Material * material2 = new Material;
        material2->kd = vec3(0.2f, 0.2f, 0);
        material2->ks = vec3(4, 4, 4);
        material2->ka = vec3(0.1f, 0.1f, 0.1f);
        material2->shininess = 10000;


        Texture * piramisTexture = new PiramisTexture(256, 256);

        Geometry * piramis = new Piramis();
        Geometry * floor = new Quad();

        tank = new Tank(vec3(-1, -7, -1),phongShader,material2);

        for(int i = 0; i<tank->getLeft()->getQuads().size(); i++){
            leftShoe.push_back(tank->getLeft()->getQuads().at(i));
        }
        for(int i = 0; i<tank->getRight()->getQuads().size(); i++){
            rightShoe.push_back(tank->getRight()->getQuads().at(i));
        }

        Object * floorObject = new Object(phongShader, material1, floor);
        floorObject->translation = vec3(-500, -8, -70);
        floorObject->scale = vec3(10000, 10000, 10000);
        //floorObject->rotationAxis = vec3(1, 0, 0);
        //floorObject->rotationAngle = (float)M_PI / 180.0f;
        objects.push_back(floorObject);


        Object * piramisObject = new Object(phongShader, material0,piramis);
        piramisObject->translation = vec3(-10, -9, -5);
        piramisObject->scale = vec3(3, 3, 3);
        objects.push_back(piramisObject);
        Object * piramisObject2 = new Object(phongShader, material0,piramis);
        piramisObject2->translation = vec3(7, -9, -6);
        piramisObject2->scale = vec3(2, 2, 2);
        objects.push_back(piramisObject2);
        Object * piramisObject3 = new Object(phongShader, material0,piramis);
        piramisObject3->translation = vec3(10, -9, -4);
        piramisObject3->scale = vec3(4, 4, 4);
        objects.push_back(piramisObject3);

        camera.wEye = tank->getCenter()-tank->getH()*5;
        camera.wLookat = tank->getCenter();
        camera.wEye.y = -1;
        camera.wVup = vec3(0, 1, 0);
        lights.resize(3);
        lights[0].wLightPos = vec4(5, 5, 4, 0);
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(3, 0, 0);
        lights[1].wLightPos = vec4(5, 10, 20, 0);
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(0, 3, 0);
        lights[2].wLightPos = vec4(-5, 5, 5, 0);
        lights[2].La = vec3(0.1f, 0.1f, 0.1f);
        lights[2].Le = vec3(0, 0, 3);
    }

    void Render() {
        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        for (Object * obj : rightShoe) obj->Draw(state);
        for (Object * obj : leftShoe) obj->Draw(state);
        for (Object * obj : objects) obj->Draw(state);
    }
    void setSebR(float v){
        tank->setSebR(v);
    }
    void setSebL(float v){
        tank->setSebL(v);
    }
    void tankAnimation(float dt, float vj, float vb){
        tank->Animate(dt, vj,vb);
        vec3 h = tank->getH();
        camera.wEye = tank->getCenter()-h*5;
        camera.wLookat = tank->getCenter();
        camera.wEye.y = -1;
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
    scene.Render();
}
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();									// exchange the two buffers
}
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    //scene.tankAnimation(key);

    if(key=='o'){ //jobb talp elore
        scene.setSebR(1);
    }
    if(key=='l'){ //jobb talp hatra
        scene.setSebR(-1);
    }
    if(key=='q'){ //bal talp elore
        scene.setSebL(1);
    }
    if(key=='a'){ //bal talp hatra
        scene.setSebL(-1);
    }
}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

    if(key=='o'){ //jobb talp elore
        scene.setSebR(0);
    }
    if(key=='l'){ //jobb talp hatra
        scene.setSebR(0);
    }
    if(key=='q'){ //bal talp elore
        scene.setSebL(0);
    }
    if(key=='a'){ //bal talp hatra
        scene.setSebL(0);
    }
}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
// Idle event indicating that some time elapsed: do animation here
void onIdle() {

    static float tend = 0;
    const float dt = 0.1f; // dt is ?infinitesimal?
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.tankAnimation( Dt, scene.getTank()->getRight()->getSeb(),scene.getTank()->getLeft()->getSeb());
    }
    glutPostRedisplay();
}