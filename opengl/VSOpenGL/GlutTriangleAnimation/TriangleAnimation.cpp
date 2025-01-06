#include <GL/glut.h>

// Global variables for the triangle colors and animation state
float color[3][3] = {
    { 1.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0 }
};
float colorIncrement = 0.01;
int dir[] = { 1, 0, 0 };

// Clears the current window and draws a triangle.
void display() {

    // Set every pixel in the frame buffer to the current clear color.
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw the triangle
    glBegin(GL_POLYGON);
    glColor3fv(color[0]); glVertex3f(-0.6, -0.75, 0.5);
    glColor3fv(color[1]); glVertex3f(0.6, -0.75, 0);
    glColor3fv(color[2]); glVertex3f(0, 0.75, 0);
    glEnd();

    // Flush drawing command buffer to make drawing happen as soon as possible.
    glFlush();
}

// Timer function to update colors
void timer(int value) {
    // Update colors
    for (int i = 0;i < 3;i++) {
        for (int j = 0;j < 3;j++) {
            color[i][j] += dir[j] * colorIncrement;
        }
    }

    // Reverse direction if colors are out of bounds
    if (color[0][0] >= 1.0f || color[0][0] <= 0.0f) {
        dir[0] = -dir[0];
        dir[1] = 1;
    }

    if (color[0][1] >= 1.0f || color[0][1] <= 0.0f) {
        dir[1] = -dir[1];
    }

    // Ensure colors stay within bounds [0, 1]
    for (int i = 0; i < 3; i++) {
        for (int j = 0;j < 3;j++) {
            if (color[i][j] > 1.0f) color[i][j] = 1.0f;
            if (color[i][j] < 0.0f) color[i][j] = 0.0f;
        }
    }

    // Redisplay the scene with the new colors
    glutPostRedisplay();

    // Register the timer function again
    glutTimerFunc(5, timer, 0);
}

// Initializes GLUT, the display mode, and main window; registers callbacks;
// enters the main event loop.
int main(int argc, char** argv) {

    // Use a single buffered window in RGB mode.
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    // Position window at (80,80)-(480,380) and give it a title.
    glutInitWindowPosition(80, 80);
    glutInitWindowSize(400, 300);
    glutCreateWindow("A Simple Triangle");

    // Register display and timer functions
    glutDisplayFunc(display);
    glutTimerFunc(0, timer, 0);

    // Enter the main event loop.
    glutMainLoop();
}
