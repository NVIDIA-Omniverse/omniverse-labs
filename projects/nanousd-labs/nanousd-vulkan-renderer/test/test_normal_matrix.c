// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* test_normal_matrix.c — validates the skinning normal-matrix math used by
 * apply_usdskel_skinning() in src/scene.c (task #14: skinned normals must use
 * the inverse-transpose of the joint matrix, not the joint matrix itself).
 *
 * build_normal_matrix / invert_affine_m4d_rowvec / xform_dir below MIRROR
 * src/scene.c verbatim (kept in sync; the helper is 8 lines). They are checked
 * here against TWO independent references so the test is not circular:
 *   (1) a Gauss-Jordan 3x3 inverse + transpose (scene.c uses cofactor/adjugate),
 *   (2) the defining geometric property: if n . t == 0 (normal perp to a surface
 *       tangent), then after deformation n' . t' == 0.
 * The end-to-end render test exercises the real scene.c wiring on a skinned asset.
 *
 * Crucially these cases use NON-UNIFORM scale + shear: a bit-identical rigid
 * render proves nothing here, because the fix is a no-op on rigid joints.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- mirrored from src/scene.c (keep in sync) ---- */
static void xform_dir(const double m[16], const float v[3], double out[3]) {
    double x = (double)v[0], y = (double)v[1], z = (double)v[2];
    out[0] = m[0] * x + m[4] * y + m[8] * z;
    out[1] = m[1] * x + m[5] * y + m[9] * z;
    out[2] = m[2] * x + m[6] * y + m[10] * z;
}
static int invert_affine_m4d_rowvec(const double m[16], double out[16]) {
    double a00=m[0],a01=m[1],a02=m[2], a10=m[4],a11=m[5],a12=m[6], a20=m[8],a21=m[9],a22=m[10];
    double c00= a11*a22-a12*a21, c01=-(a10*a22-a12*a20), c02= a10*a21-a11*a20;
    double c10=-(a01*a22-a02*a21), c11= a00*a22-a02*a20, c12=-(a00*a21-a01*a20);
    double c20= a01*a12-a02*a11, c21=-(a00*a12-a02*a10), c22= a00*a11-a01*a10;
    double det=a00*c00+a01*c01+a02*c02;
    if (fabs(det) < 1e-12) return 0;
    double id=1.0/det;
    out[0]=c00*id; out[1]=c10*id; out[2]=c20*id; out[3]=0.0;
    out[4]=c01*id; out[5]=c11*id; out[6]=c21*id; out[7]=0.0;
    out[8]=c02*id; out[9]=c12*id; out[10]=c22*id; out[11]=0.0;
    double tx=m[12],ty=m[13],tz=m[14];
    out[12]=-(tx*out[0]+ty*out[4]+tz*out[8]);
    out[13]=-(tx*out[1]+ty*out[5]+tz*out[9]);
    out[14]=-(tx*out[2]+ty*out[6]+tz*out[10]);
    out[15]=1.0;
    return 1;
}
static int build_normal_matrix(const double affine[16], double out[16]) {
    double inv[16];
    if (!invert_affine_m4d_rowvec(affine, inv)) { memcpy(out, affine, 16u*sizeof(double)); return 0; }
    for (int r=0;r<4;r++) for (int c=0;c<4;c++) out[r*4+c]=inv[c*4+r];
    return 1;
}

/* ---- independent oracle: Gauss-Jordan 3x3 inverse, then transpose ---- */
static int gj_inverse_transpose3(const double m[16], double nt[3][3]) {
    double a[3][6];
    for (int r=0;r<3;r++) {
        a[r][0]=m[r*4+0]; a[r][1]=m[r*4+1]; a[r][2]=m[r*4+2];
        a[r][3]=(r==0); a[r][4]=(r==1); a[r][5]=(r==2);
    }
    for (int col=0; col<3; col++) {
        int piv=col; for (int r=col+1;r<3;r++) if (fabs(a[r][col])>fabs(a[piv][col])) piv=r;
        if (fabs(a[piv][col])<1e-12) return 0;
        if (piv!=col) for (int k=0;k<6;k++){ double t=a[col][k]; a[col][k]=a[piv][k]; a[piv][k]=t; }
        double d=a[col][col]; for (int k=0;k<6;k++) a[col][k]/=d;
        for (int r=0;r<3;r++) if (r!=col){ double f=a[r][col]; for (int k=0;k<6;k++) a[r][k]-=f*a[col][k]; }
    }
    /* inverse is a[:,3:6]; transpose into nt */
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) nt[c][r]=a[r][3+c];
    return 1;
}

/* ---- matrix builders (row-vector convention, row-major store) ---- */
static void mat_mul(const double A[16], const double B[16], double O[16]) {
    for (int i=0;i<4;i++) for (int j=0;j<4;j++){ double s=0; for (int k=0;k<4;k++) s+=A[i*4+k]*B[k*4+j]; O[i*4+j]=s; }
}
static void mat_rotz(double d, double M[16]) {
    double c=cos(d*M_PI/180.0), s=sin(d*M_PI/180.0);
    double R[16]={c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1}; memcpy(M,R,sizeof R);
}
static void mat_roty(double d, double M[16]) {
    double c=cos(d*M_PI/180.0), s=sin(d*M_PI/180.0);
    double R[16]={c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1}; memcpy(M,R,sizeof R);
}
static void mat_scale(double x,double y,double z,double M[16]){ double S[16]={x,0,0,0,0,y,0,0,0,0,z,0,0,0,0,1}; memcpy(M,S,sizeof S);}
static void mat_trans(double x,double y,double z,double M[16]){ double T[16]={1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1}; memcpy(M,T,sizeof T);}

static int g_fail = 0;
static void check(const char* what, double got, double want, double tol) {
    if (fabs(got-want) > tol) { printf("  FAIL %s: got %.9f want %.9f (tol %.1e)\n", what, got, want, tol); g_fail++; }
}

/* For affine A, assert build_normal_matrix matches the GJ inverse-transpose for a
 * battery of normals, and that orthogonality is preserved under deformation. */
static void exercise(const char* label, const double A[16]) {
    double N[16]; build_normal_matrix(A, N);
    double nt[3][3];
    if (!gj_inverse_transpose3(A, nt)) { printf("  (%s singular — skipped)\n", label); return; }
    const float ns[5][3] = {{1,0,0},{0,1,0},{0,0,1},{0.3f,-0.7f,0.65f},{-0.5f,0.2f,0.84f}};
    for (int i=0;i<5;i++) {
        double got[3]; xform_dir(N, ns[i], got);
        double want[3] = {
            ns[i][0]*nt[0][0]+ns[i][1]*nt[1][0]+ns[i][2]*nt[2][0],
            ns[i][0]*nt[0][1]+ns[i][1]*nt[1][1]+ns[i][2]*nt[2][1],
            ns[i][0]*nt[0][2]+ns[i][1]*nt[1][2]+ns[i][2]*nt[2][2],
        };
        for (int k=0;k<3;k++) { char b[96]; snprintf(b,sizeof b,"%s invT n%d[%d]",label,i,k); check(b,got[k],want[k],1e-9); }
    }
    /* Orthogonality: n=(0,0,1), two tangents in its plane. Tangents deform by A's
     * upper-3x3 (xform_dir(A,.)); the skinned normal by N. n'.t' must stay 0. */
    float n[3]={0,0,1}; float t1[3]={1,0,0}, t2[3]={0,1,0};
    double np[3], tp1[3], tp2[3];
    xform_dir(N,n,np); xform_dir(A,t1,tp1); xform_dir(A,t2,tp2);
    char b1[64],b2[64]; snprintf(b1,sizeof b1,"%s ortho n'.t1'",label); snprintf(b2,sizeof b2,"%s ortho n'.t2'",label);
    check(b1, np[0]*tp1[0]+np[1]*tp1[1]+np[2]*tp1[2], 0.0, 1e-9);
    check(b2, np[0]*tp2[0]+np[1]*tp2[1]+np[2]*tp2[2], 0.0, 1e-9);
}

int main(void) {
    double R[16], S[16], T[16], tmp[16], A[16];

    /* (1) rigid (pure rotation): inverse-transpose == rotation, so a rigid joint
     *     leaves normals unchanged vs the old direct-apply path. */
    mat_roty(37.0, R);
    {
        double Nr[16]; build_normal_matrix(R, Nr);
        const float n[3]={0.3f,-0.7f,0.65f};
        double via_N[3], via_R[3]; xform_dir(Nr,n,via_N); xform_dir(R,n,via_R);
        for (int k=0;k<3;k++) { char b[64]; snprintf(b,sizeof b,"rigid unchanged[%d]",k); check(b,via_N[k],via_R[k],1e-9); }
    }
    exercise("rotation", R);

    /* (2) pure non-uniform scale: normal matrix is the reciprocal diagonal. */
    mat_scale(2.0, 1.0, 0.5, S);
    {
        double Ns[16]; build_normal_matrix(S, Ns);
        const float nx[3]={1,0,0}, nz[3]={0,0,1}; double ox[3], oz[3];
        xform_dir(Ns,nx,ox); xform_dir(Ns,nz,oz);
        check("scale invX", ox[0], 0.5, 1e-9);   /* 1/2 */
        check("scale invZ", oz[2], 2.0, 1e-9);   /* 1/0.5 */
    }
    exercise("scale", S);

    /* (3) composite rotation * non-uniform scale * translation (the skin case). */
    mat_scale(2.0,1.0,0.5,S); mat_roty(37.0,R); mat_trans(3,-2,5,T);
    mat_mul(R,S,tmp); mat_mul(tmp,T,A);
    exercise("rot*scale*trans", A);

    /* (4) shear (non-orthogonal upper-3x3). */
    double Sh[16]={1,0.4,0,0, 0,1,0.25,0, 0.1,0,1,0, 0,0,0,1};
    exercise("shear", Sh);

    /* (5) another composite: rotz * anisotropic scale. */
    mat_rotz(-61.0,R); mat_scale(0.3,1.7,1.0,S); mat_mul(R,S,A);
    exercise("rotz*aniso", A);

    if (g_fail == 0) { printf("test_normal_matrix: PASS (all cases)\n"); return 0; }
    printf("test_normal_matrix: FAIL (%d checks)\n", g_fail);
    return 1;
}
