// fbxio: a thin pybind11 binding over the Autodesk FBX SDK for the FBX extension's worker (adapters/fbx/worker.py).
//
// Only primitives live here: open or create a scene, list its nodes, evaluate transforms, read and write meshes,
// cameras, skeletons, skins and blend shapes, save. What they mean for Lab2Shot (which items a file holds, units and
// axes, our camera convention) is decided in Python. Matrices cross the boundary as numpy [4,4] column-vector
// matrices (translation in [:3, 3]); the SDK stores them transposed (translation in its fourth row), and its matrix
// product composes the same way (parent * child). Time crosses as frame numbers at the scene's frame rate.
//
// Built at install time by build.py against the SDK the user installed (the manual download "fbx_sdk", extension.py),
// linked statically; the SDK's libxml2 2.x comes from the extension's conda environment.

#include <fbxsdk.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<long long, py::array::c_style | py::array::forcecast>;

namespace {

// ------------------------------------------------------------------ conversions

py::array_t<double> to_numpy(const FbxAMatrix& m) {
    py::array_t<double> out({4, 4});
    auto r = out.mutable_unchecked<2>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r(i, j) = m.Get(j, i);  // stored transposed
    return out;
}

FbxAMatrix to_fbx(const double* column_vector) {  // 16 values, row-major of our column-vector matrix
    double stored[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) stored[j * 4 + i] = column_vector[i * 4 + j];  // the SDK stores it transposed
    FbxAMatrix a;
    std::memcpy(static_cast<double*>(a), stored, sizeof(stored));  // FbxAMatrix has no raw setter
    return a;
}

FbxAMatrix matrix_arg(const Array& a) {
    if (a.ndim() != 2 || a.shape(0) != 4 || a.shape(1) != 4) throw std::invalid_argument("matrix must be 4x4");
    return to_fbx(a.data());
}

std::string axis_letter(int axis) { return std::string(1, "xyz"[axis]); }

}  // namespace

// ------------------------------------------------------------------ the scene

class Scene {
public:
    explicit Scene(FbxManager* manager, FbxScene* scene) : manager_(manager), scene_(scene) { collect(); }
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;
    ~Scene() {
        if (manager_) manager_->Destroy();
    }

    // --- open / create

    // The file is a user's upload: it is read as FBX and nothing else, and reading it writes nothing. Left to itself the
    // SDK picks a reader by the file (its OBJ reader follows mtllib to any path on the server; .dxf, .3ds and others
    // have readers too) and unpacks the media embedded in an FBX into <name>.fbm/ beside it.
    static std::unique_ptr<Scene> open(const std::string& path) {
        FbxManager* manager = FbxManager::Create();
        FbxIOSettings* ios = FbxIOSettings::Create(manager, IOSROOT);
        ios->SetBoolProp(IMP_FBX_EXTRACT_EMBEDDED_DATA, false);
        manager->SetIOSettings(ios);
        const int fbx_reader = manager->GetIOPluginRegistry()->FindReaderIDByExtension("fbx");
        FbxImporter* importer = FbxImporter::Create(manager, "");
        if (fbx_reader < 0 || !importer->Initialize(path.c_str(), fbx_reader, ios)) {
            std::string why = importer->GetStatus().GetErrorString();
            manager->Destroy();
            throw std::runtime_error("cannot open: " + why);
        }
        FbxScene* scene = FbxScene::Create(manager, "scene");
        if (!importer->Import(scene)) {
            std::string why = importer->GetStatus().GetErrorString();
            manager->Destroy();
            throw std::runtime_error("cannot import: " + why);
        }
        importer->Destroy();
        // which take is read is the reader's decision (worker.py choose_take over takes() and set_take()), not the
        // file's first one (Mixamo's first, 「Take 001」, is empty)
        return std::make_unique<Scene>(manager, scene);
    }

    static std::unique_ptr<Scene> create(double fps, double start, double stop) {
        FbxManager* manager = FbxManager::Create();
        manager->SetIOSettings(FbxIOSettings::Create(manager, IOSROOT));
        FbxScene* scene = FbxScene::Create(manager, "scene");
        auto out = std::make_unique<Scene>(manager, scene);
        FbxGlobalSettings& gs = scene->GetGlobalSettings();
        gs.SetAxisSystem(FbxAxisSystem(FbxAxisSystem::eMayaYUp));
        gs.SetSystemUnit(FbxSystemUnit::cm);
        FbxTime::EMode mode = FbxTime::ConvertFrameRateToTimeMode(fps);
        gs.SetTimeMode(mode);
        if (mode == FbxTime::eCustom) gs.SetCustomFrameRate(fps);
        out->fps_ = fps;
        FbxTimeSpan span(out->time(start), out->time(stop));
        gs.SetTimelineDefaultTimeSpan(span);
        FbxAnimStack* stack = FbxAnimStack::Create(scene, "Take 001");
        stack->SetLocalTimeSpan(span);
        stack->SetReferenceTimeSpan(span);
        out->layer_ = FbxAnimLayer::Create(scene, "BaseLayer");
        stack->AddMember(out->layer_);
        scene->SetCurrentAnimationStack(stack);
        return out;
    }

    // --- facts about the file

    py::dict info() {
        FbxGlobalSettings& gs = scene_->GetGlobalSettings();
        FbxAxisSystem axes = gs.GetAxisSystem();
        int up_sign = 1, front_sign = 1;
        int up = static_cast<int>(axes.GetUpVector(up_sign)) - 1;  // 0 x, 1 y, 2 z
        FbxAxisSystem::EFrontVector parity = axes.GetFrontVector(front_sign);
        FbxTimeSpan span;
        FbxAnimStack* stack = scene_->GetCurrentAnimationStack();
        if (stack) span = stack->GetLocalTimeSpan();
        if (!stack || span.GetDuration() == FbxTime(0)) gs.GetTimelineDefaultTimeSpan(span);
        py::dict d;
        d["up"] = axis_letter(up);
        d["up_sign"] = up_sign;
        d["front_parity"] = parity == FbxAxisSystem::eParityEven ? "even" : "odd";
        d["front_sign"] = front_sign;
        d["right_handed"] = axes.GetCoorSystem() == FbxAxisSystem::eRightHanded;
        d["unit_cm"] = gs.GetSystemUnit().GetScaleFactor();
        d["fps"] = fps();
        d["start"] = frame(span.GetStart());
        d["stop"] = frame(span.GetStop());
        FbxDocumentInfo* doc = scene_->GetDocumentInfo();
        d["creator"] = doc ? std::string(doc->Original_ApplicationName.Get().Buffer()) : std::string();
        return d;
    }

    // --- nodes

    py::list nodes() {
        py::list out;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            FbxNode* n = nodes_[i];
            py::dict d;
            d["id"] = static_cast<int>(i);
            d["name"] = std::string(n->GetName());
            d["parent"] = parent_id(n);
            d["kind"] = kind(n);
            std::string skel;
            if (n->GetSkeleton()) {
                switch (n->GetSkeleton()->GetSkeletonType()) {
                    case FbxSkeleton::eRoot: skel = "root"; break;
                    case FbxSkeleton::eLimb: skel = "limb"; break;
                    case FbxSkeleton::eLimbNode: skel = "limb_node"; break;
                    case FbxSkeleton::eEffector: skel = "effector"; break;
                }
            }
            d["skeleton"] = skel;
            out.append(d);
        }
        return out;
    }

    // Global (local-to-world) matrices at frames: every parent, pivot and pre/post rotation composed by the SDK;
    // `aim`: a node with a look-at target turned to it as the SDK aims it (its +X axis at the target: a camera's).
    py::array_t<double> global_matrices(int id, const Array& frames, bool aim) {
        FbxNode* n = node(id);
        py::array_t<double> out({static_cast<py::ssize_t>(frames.size()), py::ssize_t(4), py::ssize_t(4)});
        auto o = out.mutable_unchecked<3>();
        for (py::ssize_t f = 0; f < frames.size(); ++f) {
            FbxAMatrix m = n->EvaluateGlobalTransform(time(frames.data()[f]), FbxNode::eSourcePivot, aim);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) o(f, i, j) = m.Get(j, i);
        }
        return out;
    }

    // The node's geometric transform: it moves its attribute (the mesh's points) but not its children.
    py::array_t<double> geometric(int id) {
        FbxNode* n = node(id);
        FbxAMatrix m(n->GetGeometricTranslation(FbxNode::eSourcePivot), n->GetGeometricRotation(FbxNode::eSourcePivot),
                     n->GetGeometricScaling(FbxNode::eSourcePivot));
        return to_numpy(m);
    }

    // Frames of every key on the node's own animated properties: translation, rotation, scaling, a camera's lens,
    // the blend-shape weights of its mesh (sorted, unique, as frame numbers that may be fractional).
    std::vector<double> key_frames(int id) {
        std::set<double> found;
        for (FbxAnimLayer* layer : layers())
            for (FbxProperty& p : props_of(node(id))) keys_of(p, layer, found);
        return std::vector<double>(found.begin(), found.end());
    }

    // --- takes (animation stacks)

    // Every take of the file, in the file's order: its name, its span in frames, and how many distinct key frames its
    // layers hold over every node's animated properties (the ones key_frames reads). An empty take has 0 keys.
    py::list takes() {
        py::list out;
        for (int i = 0; i < scene_->GetSrcObjectCount<FbxAnimStack>(); ++i) {
            FbxAnimStack* stack = scene_->GetSrcObject<FbxAnimStack>(i);
            FbxTimeSpan span = stack->GetLocalTimeSpan();
            std::set<double> found;
            for (FbxAnimLayer* layer : stack_layers(stack))
                for (FbxNode* n : nodes_)
                    for (FbxProperty& p : props_of(n)) keys_of(p, layer, found);
            py::dict d;
            d["name"] = std::string(stack->GetName());
            d["start"] = frame(span.GetStart());
            d["stop"] = frame(span.GetStop());
            d["keys"] = static_cast<int>(found.size());
            out.append(d);
        }
        return out;
    }

    // Make a take current: info(), key_frames() and every evaluation read it from then on.
    void set_take(const std::string& name) {
        FbxAnimStack* stack = scene_->FindSrcObject<FbxAnimStack>(name.c_str());
        if (!stack) throw std::out_of_range("no take " + name);
        scene_->SetCurrentAnimationStack(stack);
    }

    // A new take to write into (with its own base layer), current from now on: set_transform keys go into it.
    void add_take(const std::string& name, double start, double stop) {
        FbxTimeSpan span(time(start), time(stop));
        FbxAnimStack* stack = FbxAnimStack::Create(scene_, name.c_str());
        stack->SetLocalTimeSpan(span);
        stack->SetReferenceTimeSpan(span);
        layer_ = FbxAnimLayer::Create(scene_, (name + " BaseLayer").c_str());
        stack->AddMember(layer_);
        scene_->SetCurrentAnimationStack(stack);
    }

    // --- meshes

    py::dict mesh(int id) {
        FbxMesh* m = node(id)->GetMesh();
        if (!m) throw std::invalid_argument("node has no mesh");
        int nv = m->GetControlPointsCount(), np = m->GetPolygonCount();
        py::array_t<double> points({py::ssize_t(nv), py::ssize_t(3)});
        auto p = points.mutable_unchecked<2>();
        const FbxVector4* cp = m->GetControlPoints();
        for (int i = 0; i < nv; ++i)
            for (int k = 0; k < 3; ++k) p(i, k) = cp[i][k];
        std::vector<int> counts(np), indices;
        indices.reserve(m->GetPolygonVertexCount());
        for (int f = 0; f < np; ++f) {
            counts[f] = m->GetPolygonSize(f);
            for (int c = 0; c < counts[f]; ++c) indices.push_back(m->GetPolygonVertex(f, c));
        }
        py::dict d;
        d["points"] = points;
        d["counts"] = py::array(py::cast(counts));
        d["indices"] = py::array(py::cast(indices));
        d["uv"] = py::none();
        d["uv_indices"] = py::none();
        d["normals"] = py::none();
        if (FbxGeometryElementUV* uv = m->GetElementUV(0)) {
            auto& direct = uv->GetDirectArray();
            py::array_t<double> vals({py::ssize_t(direct.GetCount()), py::ssize_t(2)});
            auto v = vals.mutable_unchecked<2>();
            for (int i = 0; i < direct.GetCount(); ++i) {
                v(i, 0) = direct.GetAt(i)[0];
                v(i, 1) = direct.GetAt(i)[1];
            }
            bool by_corner = uv->GetMappingMode() == FbxGeometryElement::eByPolygonVertex;
            bool by_point = uv->GetMappingMode() == FbxGeometryElement::eByControlPoint;
            if (by_corner || by_point) {
                bool indexed = uv->GetReferenceMode() != FbxGeometryElement::eDirect;
                std::vector<int> idx(indices.size());
                for (size_t k = 0; k < indices.size(); ++k) {
                    int at = by_corner ? static_cast<int>(k) : indices[k];
                    idx[k] = indexed ? uv->GetIndexArray().GetAt(at) : at;
                }
                d["uv"] = vals;
                d["uv_indices"] = py::array(py::cast(idx));
            }
        }
        if (m->GetElementNormal(0)) {
            py::array_t<double> normals({py::ssize_t(indices.size()), py::ssize_t(3)});
            auto nn = normals.mutable_unchecked<2>();
            size_t k = 0;
            for (int f = 0; f < np; ++f)
                for (int c = 0; c < counts[f]; ++c, ++k) {
                    FbxVector4 v;
                    m->GetPolygonVertexNormal(f, c, v);
                    for (int a = 0; a < 3; ++a) nn(k, a) = v[a];
                }
            d["normals"] = normals;
        }
        return d;
    }

    // The mesh's skin: its clusters (the joint node each follows, the control points and weights it moves, the
    // mesh's and the joint's global matrices at bind time), or None.
    py::object skin(int id) {
        FbxMesh* m = node(id)->GetMesh();
        if (!m || m->GetDeformerCount(FbxDeformer::eSkin) == 0) return py::none();
        FbxSkin* s = static_cast<FbxSkin*>(m->GetDeformer(0, FbxDeformer::eSkin));
        py::list clusters;
        for (int c = 0; c < s->GetClusterCount(); ++c) {
            FbxCluster* cl = s->GetCluster(c);
            if (!cl->GetLink()) continue;
            int n = cl->GetControlPointIndicesCount();
            std::vector<int> idx(cl->GetControlPointIndices(), cl->GetControlPointIndices() + n);
            std::vector<double> w(cl->GetControlPointWeights(), cl->GetControlPointWeights() + n);
            FbxAMatrix t, tl;
            cl->GetTransformMatrix(t);
            cl->GetTransformLinkMatrix(tl);
            py::dict d;
            d["link"] = index_of(cl->GetLink());
            d["indices"] = py::array(py::cast(idx));
            d["weights"] = py::array(py::cast(w));
            d["transform"] = to_numpy(t);
            d["transform_link"] = to_numpy(tl);
            clusters.append(d);
        }
        static const char* kinds[] = {"rigid", "linear", "dual_quaternion", "blend"};
        py::dict out;
        out["type"] = kinds[std::clamp(static_cast<int>(s->GetSkinningType()), 0, 3)];
        out["clusters"] = clusters;
        return out;
    }

    // The mesh's blend shapes: one per channel, its name and the control points of its full-weight target.
    py::list blend_shapes(int id) {
        py::list out;
        FbxMesh* m = node(id)->GetMesh();
        if (!m) return out;
        for (FbxBlendShapeChannel* ch : channels(m)) {
            int nt = ch->GetTargetShapeCount();
            if (nt == 0) continue;
            FbxShape* shape = ch->GetTargetShape(nt - 1);
            int nv = shape->GetControlPointsCount();
            py::array_t<double> pts({py::ssize_t(nv), py::ssize_t(3)});
            auto p = pts.mutable_unchecked<2>();
            for (int i = 0; i < nv; ++i)
                for (int k = 0; k < 3; ++k) p(i, k) = shape->GetControlPointAt(i)[k];
            py::dict d;
            d["name"] = std::string(ch->GetName());
            d["points"] = pts;
            d["full_weight"] = nt > 0 ? ch->GetTargetShapeFullWeights()[nt - 1] : 100.0;
            out.append(d);
        }
        return out;
    }

    // Blend-shape weights at frames [F,B], 0-1 (the SDK's DeformPercent / 100), channels as blend_shapes lists them.
    py::array_t<double> blend_weights(int id, const Array& frames) {
        FbxMesh* m = node(id)->GetMesh();
        std::vector<FbxBlendShapeChannel*> chs;
        if (m)
            for (FbxBlendShapeChannel* ch : channels(m))
                if (ch->GetTargetShapeCount() > 0) chs.push_back(ch);
        py::array_t<double> out({static_cast<py::ssize_t>(frames.size()), static_cast<py::ssize_t>(chs.size())});
        auto o = out.mutable_unchecked<2>();
        for (py::ssize_t f = 0; f < frames.size(); ++f)
            for (size_t b = 0; b < chs.size(); ++b) o(f, b) = chs[b]->DeformPercent.EvaluateValue(time(frames.data()[f])) / 100.0;
        return out;
    }

    // --- cameras

    // The lens at frames: focal length in mm whatever drives it (the focal length itself, or a field of view through
    // the film back), the film back in mm and the resolution it records. Where it looks is its node's: an FBX camera
    // aims down its node's +X axis, up +Y (the SDK's convention; global_matrices gives the node's placement).
    // The lens centre (film offset, inches in FBX) per frame in mm, the squeeze, and the camera's lab2shot: user properties.
    py::dict camera(int id, const Array& frames) {
        FbxCamera* cam = node(id)->GetCamera();
        if (!cam) throw std::invalid_argument("node has no camera");
        py::ssize_t nf = frames.size();
        py::array_t<double> focal(nf);
        py::array_t<double> center(std::vector<py::ssize_t>{nf, 2});
        auto fo = focal.mutable_unchecked<1>();
        auto ce = center.mutable_unchecked<2>();
        for (py::ssize_t f = 0; f < nf; ++f) {
            FbxTime t = time(frames.data()[f]);
            switch (cam->GetApertureMode()) {
                case FbxCamera::eFocalLength: fo(f) = cam->FocalLength.EvaluateValue(t); break;
                case FbxCamera::eVertical: fo(f) = cam->ComputeFocalLength(cam->FieldOfViewY.EvaluateValue(t)); break;
                case FbxCamera::eHorizAndVert: fo(f) = cam->ComputeFocalLength(cam->FieldOfViewX.EvaluateValue(t)); break;
                default: fo(f) = cam->ComputeFocalLength(cam->FieldOfView.EvaluateValue(t)); break;
            }
            ce(f, 0) = cam->FilmOffsetX.EvaluateValue(t) * 25.4;
            ce(f, 1) = cam->FilmOffsetY.EvaluateValue(t) * 25.4;
        }
        py::dict d;
        d["focal_mm"] = focal;
        d["center_mm"] = center;
        d["pixel_aspect"] = cam->FilmSqueezeRatio.Get();
        d["properties"] = user_properties(cam, "lab2shot:");
        d["film_width_mm"] = cam->GetApertureWidth() * 25.4;  // FBX keeps the film back in inches
        d["film_height_mm"] = cam->GetApertureHeight() * 25.4;
        static const char* modes[] = {"window", "fixed_ratio", "fixed_resolution", "fixed_width", "fixed_height"};
        d["aspect_mode"] = modes[std::clamp(static_cast<int>(cam->GetAspectRatioMode()), 0, 4)];
        d["aspect_width"] = cam->AspectWidth.Get();
        d["aspect_height"] = cam->AspectHeight.Get();
        return d;
    }

    // The global matrices a bind pose records, node id -> [4,4] (the first bind pose of the file; empty without one).
    py::dict bind_pose() {
        py::dict out;
        for (int i = 0; i < scene_->GetPoseCount(); ++i) {
            FbxPose* pose = scene_->GetPose(i);
            if (!pose->IsBindPose()) continue;
            for (int k = 0; k < pose->GetCount(); ++k) {
                int id = index_of(pose->GetNode(k));
                if (id < 0) continue;
                FbxMatrix m = pose->GetMatrix(k);
                FbxAMatrix a;
                std::memcpy(static_cast<double*>(a), static_cast<double*>(m), sizeof(double) * 16);
                out[py::int_(id)] = to_numpy(a);
            }
            break;
        }
        return out;
    }

    // ------------------------------------------------------------------ writing

    // Records the file's axis system ("maya_y": Y up, right-handed; "max": Z up, right-handed, front -Y) and unit
    // (centimetres per unit) as they are, with no conversion of the scene (create() writes maya_y and cm).
    void set_axes(const std::string& system, double unit_cm) {
        FbxGlobalSettings& gs = scene_->GetGlobalSettings();
        if (system == "maya_y") gs.SetAxisSystem(FbxAxisSystem(FbxAxisSystem::eMayaYUp));
        else if (system == "max") gs.SetAxisSystem(FbxAxisSystem(FbxAxisSystem::eMax));
        else throw std::invalid_argument("axis system must be maya_y or max");
        gs.SetSystemUnit(FbxSystemUnit(unit_cm));
    }

    // A new node under `parent` (-1: the root). Returns its id.
    int add_node(int parent, const std::string& name) {
        FbxNode* n = FbxNode::Create(scene_, name.c_str());
        (parent < 0 ? scene_->GetRootNode() : node(parent))->AddChild(n);
        nodes_.push_back(n);
        return static_cast<int>(nodes_.size()) - 1;
    }

    // The node's local transform (Euler XYZ rotation): one matrix is a still value, several are keys at `frames`
    // (linear between them), the rotation unrolled so it never jumps by 360 degrees.
    void set_transform(int id, const Array& frames, const Array& matrices) {
        FbxNode* n = node(id);
        py::ssize_t nf = frames.size();
        if (matrices.ndim() != 3 || matrices.shape(0) != nf || matrices.shape(1) != 4 || matrices.shape(2) != 4)
            throw std::invalid_argument("matrices must be [F,4,4] with one per frame");
        n->SetRotationOrder(FbxNode::eSourcePivot, eEulerXYZ);
        if (nf == 1) {
            FbxAMatrix m = to_fbx(matrices.data());
            n->LclTranslation.Set(m.GetT());
            n->LclRotation.Set(m.GetR());
            n->LclScaling.Set(m.GetS());
            return;
        }
        FbxAnimCurve* curves[9];
        FbxPropertyT<FbxDouble3>* props[3] = {&n->LclTranslation, &n->LclRotation, &n->LclScaling};
        const char* comps[3] = {FBXSDK_CURVENODE_COMPONENT_X, FBXSDK_CURVENODE_COMPONENT_Y, FBXSDK_CURVENODE_COMPONENT_Z};
        for (int p = 0; p < 3; ++p) {
            props[p]->GetCurveNode(layer(), true);
            for (int c = 0; c < 3; ++c) {
                curves[p * 3 + c] = props[p]->GetCurve(layer(), comps[c], true);
                curves[p * 3 + c]->KeyModifyBegin();
            }
        }
        for (py::ssize_t f = 0; f < nf; ++f) {
            FbxAMatrix m = to_fbx(matrices.data() + f * 16);
            FbxVector4 trs[3] = {m.GetT(), m.GetR(), m.GetS()};
            FbxTime t = time(frames.data()[f]);
            for (int p = 0; p < 3; ++p)
                for (int c = 0; c < 3; ++c) {
                    FbxAnimCurve* cv = curves[p * 3 + c];
                    int k = cv->KeyAdd(t);
                    cv->KeySet(k, t, static_cast<float>(trs[p][c]), FbxAnimCurveDef::eInterpolationLinear);
                }
            if (f == 0) {
                n->LclTranslation.Set(trs[0]);
                n->LclRotation.Set(trs[1]);
                n->LclScaling.Set(trs[2]);
            }
        }
        for (FbxAnimCurve* cv : curves) cv->KeyModifyEnd();
        FbxAnimCurveFilterUnroll unroll;
        FbxAnimCurve* rotation[3] = {curves[3], curves[4], curves[5]};
        unroll.Apply(rotation, 3);
    }

    // The node's mesh: control points [V,3], corners per face, the corners' points, optional face-varying UVs
    // (values [U,2] + an index per corner) and normals (one per corner).
    void set_mesh(int id, const Array& points, const IntArray& counts, const IntArray& indices, py::object uv,
                  py::object uv_indices, py::object normals) {
        FbxNode* n = node(id);
        FbxMesh* m = FbxMesh::Create(scene_, (std::string(n->GetName()) + "Shape").c_str());
        py::ssize_t nv = points.shape(0);
        m->InitControlPoints(static_cast<int>(nv));
        FbxVector4* cp = m->GetControlPoints();
        for (py::ssize_t i = 0; i < nv; ++i) cp[i] = FbxVector4(points.at(i, 0), points.at(i, 1), points.at(i, 2));
        size_t k = 0;
        for (py::ssize_t f = 0; f < counts.size(); ++f) {
            m->BeginPolygon();
            for (long long c = 0; c < counts.data()[f]; ++c) m->AddPolygon(static_cast<int>(indices.data()[k++]));
            m->EndPolygon();
        }
        if (!uv.is_none() && !uv_indices.is_none()) {
            Array vals = uv.cast<Array>();
            IntArray idx = uv_indices.cast<IntArray>();
            FbxGeometryElementUV* e = m->CreateElementUV("map1");
            e->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
            e->SetReferenceMode(FbxGeometryElement::eIndexToDirect);
            for (py::ssize_t i = 0; i < vals.shape(0); ++i) e->GetDirectArray().Add(FbxVector2(vals.at(i, 0), vals.at(i, 1)));
            for (py::ssize_t i = 0; i < idx.size(); ++i) e->GetIndexArray().Add(static_cast<int>(idx.data()[i]));
        }
        if (!normals.is_none()) {
            Array nn = normals.cast<Array>();
            FbxGeometryElementNormal* e = m->CreateElementNormal();
            e->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
            e->SetReferenceMode(FbxGeometryElement::eDirect);
            for (py::ssize_t i = 0; i < nn.shape(0); ++i) e->GetDirectArray().Add(FbxVector4(nn.at(i, 0), nn.at(i, 1), nn.at(i, 2)));
        }
        n->SetNodeAttribute(m);
    }

    // The node's camera: focal length in mm (one value, or a key per frame), the film back in mm (stored in inches),
    // driven by its focal length, with a fixed resolution in pixels; the lens centre [F,2] or [1,2] in mm (film offset),
    // the pixel aspect (squeeze) and named user properties (set_user_properties). It looks down its node's +X axis, up +Y.
    void set_camera(int id, const Array& frames, const Array& focal_mm, double film_width_mm, double film_height_mm, int width,
                    int height, const Array& center_mm, double pixel_aspect, const py::dict& properties) {
        FbxNode* n = node(id);
        FbxCamera* cam = FbxCamera::Create(scene_, (std::string(n->GetName()) + "Shape").c_str());
        cam->SetApertureFormat(FbxCamera::eCustomAperture);
        cam->SetApertureMode(FbxCamera::eFocalLength);
        cam->SetApertureWidth(film_width_mm / 25.4);
        cam->SetApertureHeight(film_height_mm / 25.4);
        if (width > 0 && height > 0) cam->SetAspect(FbxCamera::eFixedResolution, width, height);
        cam->FilmSqueezeRatio.Set(pixel_aspect);
        n->SetNodeAttribute(cam);
        std::vector<double> focal(focal_mm.data(), focal_mm.data() + focal_mm.size()), x, y;
        for (py::ssize_t f = 0; f < center_mm.shape(0); ++f) {
            x.push_back(center_mm.at(f, 0) / 25.4);
            y.push_back(center_mm.at(f, 1) / 25.4);
        }
        set_keyed(cam->FocalLength, frames, focal);
        set_keyed(cam->FilmOffsetX, frames, x);
        set_keyed(cam->FilmOffsetY, frames, y);
        set_user_properties(cam, properties);
    }

    // A value, or a linear key at each frame when the values change.
    void set_keyed(FbxProperty prop, const Array& frames, const std::vector<double>& values) {
        prop.Set(values.at(0));
        if (std::all_of(values.begin(), values.end(), [&](double v) { return v == values[0]; })) return;
        FbxAnimCurve* cv = prop.GetCurve(layer(), true);
        if (!cv) throw std::runtime_error(std::string("no animation curve for ") + prop.GetName().Buffer());
        cv->KeyModifyBegin();
        for (size_t f = 0; f < values.size(); ++f) {
            FbxTime t = time(frames.data()[f]);
            cv->KeySet(cv->KeyAdd(t), t, static_cast<float>(values[f]), FbxAnimCurveDef::eInterpolationLinear);
        }
        cv->KeyModifyEnd();
    }

    // Named user properties on an object: a text, a number, or {frames, values} as a number keyed at those frames.
    void set_user_properties(FbxObject* o, const py::dict& properties) {
        for (auto item : properties) {
            std::string name = py::str(item.first);
            py::handle v = item.second;
            bool text = py::isinstance<py::str>(v);
            FbxProperty p = FbxProperty::Create(o, text ? FbxStringDT : FbxDoubleDT, name.c_str());
            p.ModifyFlag(FbxPropertyFlags::eUserDefined, true);
            if (text) {
                p.Set(FbxString(v.cast<std::string>().c_str()));
            } else if (py::isinstance<py::dict>(v)) {
                py::dict d = v.cast<py::dict>();
                std::vector<double> fr = d["frames"].cast<std::vector<double>>(), vals = d["values"].cast<std::vector<double>>();
                p.ModifyFlag(FbxPropertyFlags::eAnimatable, true);
                Array at = py::cast(fr);
                set_keyed(p, at, vals);
            } else {
                p.Set(v.cast<double>());
            }
        }
    }

    // An object's user properties whose names start with `prefix`: texts, numbers, and a keyed number as {frames, values}.
    py::dict user_properties(FbxObject* o, const std::string& prefix) const {
        py::dict out;
        for (FbxProperty p = o->GetFirstProperty(); p.IsValid(); p = o->GetNextProperty(p)) {
            if (!p.GetFlag(FbxPropertyFlags::eUserDefined)) continue;
            std::string name = p.GetName().Buffer();
            if (name.rfind(prefix, 0) != 0) continue;
            EFbxType type = p.GetPropertyDataType().GetType();
            if (type == eFbxString) {
                out[py::str(name)] = std::string(p.Get<FbxString>().Buffer());
                continue;
            }
            if (type != eFbxDouble) continue;
            std::set<double> keys;
            for (FbxAnimLayer* l : layers()) keys_of(p, l, keys);
            if (keys.size() < 2) {
                out[py::str(name)] = keys.empty() ? p.Get<FbxDouble>() : p.EvaluateValue<FbxDouble>(time(*keys.begin()));
                continue;
            }
            py::list fr, vals;
            for (double k : keys) {
                fr.append(static_cast<long long>(std::llround(k)));
                vals.append(p.EvaluateValue<FbxDouble>(time(k)));
            }
            py::dict d;
            d["frames"] = fr;
            d["values"] = vals;
            out[py::str(name)] = d;
        }
        return out;
    }

    // Makes the node a joint: the root of a skeleton or a limb under it.
    void set_skeleton(int id, bool root, double size) {
        FbxNode* n = node(id);
        FbxSkeleton* s = FbxSkeleton::Create(scene_, (std::string(n->GetName()) + "Attr").c_str());
        s->SetSkeletonType(root ? FbxSkeleton::eRoot : FbxSkeleton::eLimbNode);
        s->Size.Set(size);
        n->SetNodeAttribute(s);
    }

    // Skins the node's mesh: per cluster (joint id, control point indices, weights, the mesh's global matrix at bind,
    // the joint's global matrix at bind), linear skinning, link mode "total one" (a point's weights add up to one).
    void add_skin(int id, const py::list& clusters) {
        FbxMesh* m = node(id)->GetMesh();
        if (!m) throw std::invalid_argument("node has no mesh");
        FbxSkin* skin = FbxSkin::Create(scene_, "");
        skin->SetSkinningType(FbxSkin::eLinear);
        for (auto item : clusters) {
            py::tuple c = item.cast<py::tuple>();
            FbxCluster* cl = FbxCluster::Create(scene_, "");
            cl->SetLink(node(c[0].cast<int>()));
            cl->SetLinkMode(FbxCluster::eTotalOne);
            IntArray idx = c[1].cast<IntArray>();
            Array w = c[2].cast<Array>();
            for (py::ssize_t i = 0; i < idx.size(); ++i) cl->AddControlPointIndex(static_cast<int>(idx.data()[i]), w.data()[i]);
            cl->SetTransformMatrix(matrix_arg(c[3].cast<Array>()));
            cl->SetTransformLinkMatrix(matrix_arg(c[4].cast<Array>()));
            skin->AddCluster(cl);
        }
        m->AddDeformer(skin);
    }

    // Adds a blend-shape channel to the node's mesh: its full-weight target's control points [V,3] and its weight
    // (0-1) per frame, keyed as the SDK's DeformPercent (one value: still).
    void add_blend_shape(int id, const std::string& name, const Array& points, const Array& frames, const Array& weights) {
        FbxMesh* m = node(id)->GetMesh();
        if (!m) throw std::invalid_argument("node has no mesh");
        FbxBlendShape* bs = m->GetDeformerCount(FbxDeformer::eBlendShape)
                                ? static_cast<FbxBlendShape*>(m->GetDeformer(0, FbxDeformer::eBlendShape))
                                : nullptr;
        if (!bs) {
            bs = FbxBlendShape::Create(scene_, (std::string(node(id)->GetName()) + "_blendShape").c_str());
            m->AddDeformer(bs);
        }
        FbxBlendShapeChannel* ch = FbxBlendShapeChannel::Create(scene_, name.c_str());
        FbxShape* shape = FbxShape::Create(scene_, name.c_str());
        py::ssize_t nv = points.shape(0);
        shape->InitControlPoints(static_cast<int>(nv));
        for (py::ssize_t i = 0; i < nv; ++i) shape->SetControlPointAt(FbxVector4(points.at(i, 0), points.at(i, 1), points.at(i, 2)), static_cast<int>(i));
        ch->AddTargetShape(shape, 100.0);
        bs->AddBlendShapeChannel(ch);
        ch->DeformPercent.Set(weights.data()[0] * 100.0);
        if (weights.size() > 1) {
            FbxAnimCurve* cv = ch->DeformPercent.GetCurve(layer(), true);
            cv->KeyModifyBegin();
            for (py::ssize_t f = 0; f < weights.size(); ++f) {
                FbxTime t = time(frames.data()[f]);
                cv->KeySet(cv->KeyAdd(t), t, static_cast<float>(weights.data()[f] * 100.0), FbxAnimCurveDef::eInterpolationLinear);
            }
            cv->KeyModifyEnd();
        }
    }

    // A bind pose: the global matrix [4,4] of each node id at bind time (the skinned meshes and their joints).
    void add_bind_pose(const std::vector<int>& ids, const Array& matrices) {
        FbxPose* pose = FbxPose::Create(scene_, "BindPose");
        pose->SetIsBindPose(true);
        for (size_t i = 0; i < ids.size(); ++i) {
            FbxAMatrix a = to_fbx(matrices.data() + i * 16);
            FbxMatrix m;
            std::memcpy(static_cast<double*>(m), static_cast<double*>(a), sizeof(double) * 16);
            pose->Add(node(ids[i]), m);
        }
        scene_->AddPose(pose);
    }

    // Writes the scene (binary FBX 7.x, the SDK's default writer, or ASCII).
    void save(const std::string& path, bool binary) {
        FbxExporter* exporter = FbxExporter::Create(manager_, "");
        int format = -1;
        if (!binary) {
            FbxIOPluginRegistry* reg = manager_->GetIOPluginRegistry();
            for (int i = 0; i < reg->GetWriterFormatCount(); ++i)
                if (reg->WriterIsFBX(i) && std::string(reg->GetWriterFormatDescription(i)).find("ascii") != std::string::npos) format = i;
        }
        if (!exporter->Initialize(path.c_str(), format, manager_->GetIOSettings()))
            throw std::runtime_error(std::string("cannot write: ") + exporter->GetStatus().GetErrorString());
        if (!exporter->Export(scene_)) throw std::runtime_error(std::string("cannot write: ") + exporter->GetStatus().GetErrorString());
        exporter->Destroy();
    }

private:
    FbxManager* manager_;
    FbxScene* scene_;
    FbxAnimLayer* layer_ = nullptr;
    double fps_ = 0.0;
    std::vector<FbxNode*> nodes_;  // every node below the root, parents first (depth-first)

    void collect() {
        nodes_.clear();
        std::vector<FbxNode*> stack;
        FbxNode* root = scene_->GetRootNode();
        for (int i = root->GetChildCount() - 1; i >= 0; --i) stack.push_back(root->GetChild(i));
        while (!stack.empty()) {
            FbxNode* n = stack.back();
            stack.pop_back();
            nodes_.push_back(n);
            for (int i = n->GetChildCount() - 1; i >= 0; --i) stack.push_back(n->GetChild(i));
        }
    }

    FbxNode* node(int id) const {
        if (id < 0 || id >= static_cast<int>(nodes_.size())) throw std::out_of_range("no node " + std::to_string(id));
        return nodes_[id];
    }

    int index_of(FbxNode* n) const {
        auto it = std::find(nodes_.begin(), nodes_.end(), n);
        return it == nodes_.end() ? -1 : static_cast<int>(it - nodes_.begin());
    }

    int parent_id(FbxNode* n) const { return n->GetParent() ? index_of(n->GetParent()) : -1; }

    static std::string kind(FbxNode* n) {
        FbxNodeAttribute* a = n->GetNodeAttribute();
        if (!a) return "null";
        switch (a->GetAttributeType()) {
            case FbxNodeAttribute::eNull: return "null";
            case FbxNodeAttribute::eMesh: return "mesh";
            case FbxNodeAttribute::eCamera: return "camera";
            case FbxNodeAttribute::eSkeleton: return "skeleton";
            case FbxNodeAttribute::eLight: return "light";
            default: return "other";
        }
    }

    double fps() const {
        if (fps_ > 0) return fps_;
        FbxGlobalSettings& gs = scene_->GetGlobalSettings();
        FbxTime::EMode mode = gs.GetTimeMode();
        return mode == FbxTime::eCustom ? gs.GetCustomFrameRate() : FbxTime::GetFrameRate(mode);
    }

    FbxTime time(double frame) const {
        FbxTime t;
        t.SetSecondDouble(frame / fps());
        return t;
    }

    double frame(const FbxTime& t) const { return t.GetSecondDouble() * fps(); }

    FbxAnimLayer* layer() {
        if (!layer_) throw std::runtime_error("scene is not being written");
        return layer_;
    }

    std::vector<FbxAnimLayer*> layers() const { return stack_layers(scene_->GetCurrentAnimationStack()); }

    static std::vector<FbxAnimLayer*> stack_layers(FbxAnimStack* stack) {
        std::vector<FbxAnimLayer*> out;
        if (!stack) return out;
        for (int i = 0; i < stack->GetMemberCount<FbxAnimLayer>(); ++i) out.push_back(stack->GetMember<FbxAnimLayer>(i));
        return out;
    }

    // A node's own animated properties: translation, rotation, scaling, a camera's lens, its mesh's blend-shape weights.
    static std::vector<FbxProperty> props_of(FbxNode* n) {
        std::vector<FbxProperty> props = {n->LclTranslation, n->LclRotation, n->LclScaling};
        if (FbxCamera* cam = n->GetCamera()) {
            props.push_back(cam->FocalLength);
            props.push_back(cam->FieldOfView);
            props.push_back(cam->FieldOfViewX);
            props.push_back(cam->FieldOfViewY);
            props.push_back(cam->FilmOffsetX);
            props.push_back(cam->FilmOffsetY);
        }
        if (FbxMesh* mesh = n->GetMesh()) {
            for (FbxBlendShapeChannel* ch : channels(mesh)) props.push_back(ch->DeformPercent);
        }
        return props;
    }

    void keys_of(FbxProperty& p, FbxAnimLayer* layer, std::set<double>& found) const {
        FbxAnimCurveNode* cn = p.GetCurveNode(layer);
        if (!cn) return;
        for (unsigned int ch = 0; ch < cn->GetChannelsCount(); ++ch)
            for (int c = 0; c < cn->GetCurveCount(ch); ++c) {
                FbxAnimCurve* cv = cn->GetCurve(ch, c);
                for (int k = 0; k < cv->KeyGetCount(); ++k) found.insert(frame(cv->KeyGetTime(k)));
            }
    }

    static std::vector<FbxBlendShapeChannel*> channels(FbxMesh* m) {
        std::vector<FbxBlendShapeChannel*> out;
        for (int d = 0; d < m->GetDeformerCount(FbxDeformer::eBlendShape); ++d) {
            FbxBlendShape* bs = static_cast<FbxBlendShape*>(m->GetDeformer(d, FbxDeformer::eBlendShape));
            for (int c = 0; c < bs->GetBlendShapeChannelCount(); ++c) out.push_back(bs->GetBlendShapeChannel(c));
        }
        return out;
    }
};

PYBIND11_MODULE(fbxio, m) {
    m.doc() = "A thin binding over the Autodesk FBX SDK for Lab2Shot's FBX worker (column-vector [4,4] matrices, frames).";
    // the binding's API version: 2 = takes() / set_take() / add_take(); 3 = a camera's lens centre, squeeze and user
    // properties (camera(), set_camera()); 4 = open() reads FBX only and unpacks no embedded media. worker.py refuses an
    // older build (E-FBX-REBUILD)
    m.attr("API") = 4;
    m.def("sdk_version", []() { return std::string(FbxManager::GetVersion(true)); }, "The FBX SDK's version.");
    m.def("open", &Scene::open, py::arg("path"), "Read an FBX file into a Scene (no take chosen: set_take() makes one current).");
    m.def("create", &Scene::create, py::arg("fps"), py::arg("start"), py::arg("stop"),
          "A new Scene to write: Maya Y-up right-handed axes, centimetres, this frame rate and time span.");
    py::class_<Scene>(m, "Scene")
        .def("info", &Scene::info, "Axis system (up, up_sign, front_parity, front_sign, right_handed), unit_cm, fps, start, stop, creator.")
        .def("nodes", &Scene::nodes, "Every node below the root, parents first: id, name, parent (-1: top), kind, skeleton type.")
        .def("global_matrices", &Scene::global_matrices, py::arg("id"), py::arg("frames"), py::arg("aim") = false,
             "Local-to-world [F,4,4] at frames (aim: turned to its look-at target, as a camera with one is).")
        .def("geometric", &Scene::geometric, py::arg("id"), "The node's geometric transform [4,4] (moves its attribute only).")
        .def("key_frames", &Scene::key_frames, py::arg("id"), "Frames of the keys on the node's TRS, lens and blend weights.")
        .def("takes", &Scene::takes, "Every take in the file's order: name, start, stop (frames), keys (distinct key frames over every node).")
        .def("set_take", &Scene::set_take, py::arg("name"), "Make a take current: info(), key_frames() and every evaluation read it.")
        .def("add_take", &Scene::add_take, py::arg("name"), py::arg("start"), py::arg("stop"), "A new take to write into, current from now on.")
        .def("mesh", &Scene::mesh, py::arg("id"), "points, counts, indices, uv + uv_indices, normals of the node's mesh.")
        .def("skin", &Scene::skin, py::arg("id"), "The mesh's skin: type and clusters (link, indices, weights, transform, transform_link).")
        .def("blend_shapes", &Scene::blend_shapes, py::arg("id"), "The mesh's blend-shape channels: name, full-weight target points.")
        .def("blend_weights", &Scene::blend_weights, py::arg("id"), py::arg("frames"), "Blend-shape weights [F,B], 0-1.")
        .def("camera", &Scene::camera, py::arg("id"), py::arg("frames"),
             "Lens (focal mm and centre mm per frame, film back mm, squeeze, lab2shot: user properties) and resolution; it aims down its node's +X, up +Y.")
        .def("bind_pose", &Scene::bind_pose, "node id -> global matrix at bind, from the file's bind pose.")
        .def("set_axes", &Scene::set_axes, py::arg("system"), py::arg("unit_cm"), "Record the axis system (maya_y / max) and unit as they are.")
        .def("add_node", &Scene::add_node, py::arg("parent"), py::arg("name"), "A new node under parent (-1: the root); its id.")
        .def("set_transform", &Scene::set_transform, py::arg("id"), py::arg("frames"), py::arg("matrices"), "Local transform: still or keyed.")
        .def("set_mesh", &Scene::set_mesh, py::arg("id"), py::arg("points"), py::arg("counts"), py::arg("indices"),
             py::arg("uv") = py::none(), py::arg("uv_indices") = py::none(), py::arg("normals") = py::none(), "The node's mesh.")
        .def("set_camera", &Scene::set_camera, py::arg("id"), py::arg("frames"), py::arg("focal_mm"), py::arg("film_width_mm"),
             py::arg("film_height_mm"), py::arg("width"), py::arg("height"), py::arg("center_mm"), py::arg("pixel_aspect"),
             py::arg("properties"), "The node's camera.")
        .def("set_skeleton", &Scene::set_skeleton, py::arg("id"), py::arg("root"), py::arg("size") = 1.0, "Make the node a joint.")
        .def("add_skin", &Scene::add_skin, py::arg("id"), py::arg("clusters"), "Skin the node's mesh.")
        .def("add_blend_shape", &Scene::add_blend_shape, py::arg("id"), py::arg("name"), py::arg("points"), py::arg("frames"),
             py::arg("weights"), "A blend-shape channel with its weight per frame.")
        .def("add_bind_pose", &Scene::add_bind_pose, py::arg("ids"), py::arg("matrices"), "A bind pose.")
        .def("save", &Scene::save, py::arg("path"), py::arg("binary") = true, "Write the FBX file.");
}
