/*
 * Regression for compact PI/native-instance RT transforms.
 *
 * SceneInstanceTransform stores USD row-vector affine data compactly with
 * translation in m[9..11]. VkTransformMatrixKHR needs that translation in the
 * 3x4 row elements 3/7/11. Dropping those slots renders compact instances at
 * the prototype origin, which showed up as an extra Ironwood trunk in Vulkan.
 */

#include "../src/scene.h"

#include <math.h>
#include <stdio.h>

static int nearf(float a, float b)
{
    return fabsf(a - b) < 1.0e-6f;
}

int main(void)
{
    SceneInstanceTransform t = {{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    }};
    float vk[12] = {0};
    const float expect[12] = {
        1.0f, 4.0f, 7.0f, 10.0f,
        2.0f, 5.0f, 8.0f, 11.0f,
        3.0f, 6.0f, 9.0f, 12.0f
    };

    scene_instance_transform_to_vk3x4(&t, vk);
    for (int i = 0; i < 12; i++) {
        if (!nearf(vk[i], expect[i])) {
            fprintf(stderr,
                    "FAIL: vk[%d] expected %.1f, got %.1f\n",
                    i, expect[i], vk[i]);
            return 1;
        }
    }

    printf("test_scene_instance_transform: PASS\n");
    return 0;
}
