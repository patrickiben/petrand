// =============================================================================
// PetRand: Sinai Billiard Chaos Deflectors & Scatterers
// Parametric OpenSCAD Model for FDM 3D Printing
// =============================================================================
// Rationale (a hypothesis to test, not a guarantee):
// Rectangular arenas allow periodic or quasi-periodic bounce paths. In the idealized
// Sinai billiard (a point particle, no friction, convex scatterers), scatterers make
// the motion chaotic and mixing. A self-propelled toy with friction, spin and its own
// motor controller is not that system, so whether these deflectors raise the entropy
// of impact timing or position has to be measured (see paper/RESEARCH_PLAN.md).
// =============================================================================

$fn = 120; // Smooth curve tessellation

// --- Parametric Dimensions (mm) ---
wall_height       = 120.0; // Height to prevent ball ejection (PetGravity ball dia = 50mm / 2.0 in)
wall_thickness    = 4.0;
corner_radius     = 60.0;  // Curved convex scatter radius
scatterer_radius  = 45.0;  // Central circular scatterer radius
screw_hole_dia    = 4.2;   // M4 mounting screw clearance

// --- Part Selector ---
// Change this variable to render individual components for STL export:
// "corner_wedge", "central_scatterer", "webcam_rig_bracket", or "all"
PART_TO_RENDER = "all";


module corner_wedge() {
    difference() {
        // Main triangular body with curved convex face
        intersection() {
            cube([corner_radius, corner_radius, wall_height]);
            cylinder(r = corner_radius, h = wall_height);
        }
        
        // Concave cut toward the arena: this face focuses rather than disperses.
        // (A dispersing, Sinai-style scatterer would be convex toward the ball.)
        translate([corner_radius, corner_radius, -1])
            cylinder(r = corner_radius * 0.75, h = wall_height + 2);

        // Mounting screw countersinks
        translate([corner_radius * 0.25, corner_radius * 0.25, -1])
            cylinder(d = screw_hole_dia, h = wall_height + 2);
    }
}


module central_scatterer() {
    difference() {
        union() {
            // Fluted cylindrical dispersal obstacle
            cylinder(r = scatterer_radius, h = wall_height * 0.7);
            
            // Fluted dispersing lobes (sinusoidal perimeter)
            for (i = [0 : 5]) {
                rotate([0, 0, i * 60])
                    translate([scatterer_radius * 0.85, 0, 0])
                        cylinder(r = scatterer_radius * 0.35, h = wall_height * 0.7);
            }
        }
        
        // Hollow interior to save filament & weight
        translate([0, 0, wall_thickness])
            cylinder(r = scatterer_radius * 0.75, h = wall_height);
            
        // Weighted ballast fill hole (fill with sand or lead shot for stability)
        translate([0, 0, -1])
            cylinder(d = 20.0, h = wall_thickness + 2);
    }
}


module webcam_rig_bracket() {
    // 1/4"-20 threaded mount bracket for overhead camera suspension
    bracket_width  = 50.0;
    bracket_length = 80.0;
    bracket_thick  = 8.0;
    
    difference() {
        cube([bracket_length, bracket_width, bracket_thick], center = true);
        
        // 1/4"-20 camera screw clearance hole (dia ~ 6.5mm)
        cylinder(d = 6.5, h = bracket_thick + 2, center = true);
        
        // Arena rim clamping slots
        translate([25, 0, 0])
            cylinder(d = screw_hole_dia, h = bracket_thick + 2, center = true);
        translate([-25, 0, 0])
            cylinder(d = screw_hole_dia, h = bracket_thick + 2, center = true);
    }
}


// --- Assembly View ---
if (PART_TO_RENDER == "corner_wedge") {
    corner_wedge();
} else if (PART_TO_RENDER == "central_scatterer") {
    central_scatterer();
} else if (PART_TO_RENDER == "webcam_rig_bracket") {
    webcam_rig_bracket();
} else {
    // Render full demonstration plate
    translate([-70, 0, 0]) corner_wedge();
    translate([70, 0, 0]) central_scatterer();
    translate([0, -100, 4]) webcam_rig_bracket();
}
