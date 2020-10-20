#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <tuple>
#include <queue>
#include <iostream>
#include "fst/fst.h"
#include "fst/script/fstscript.h"
#include "fst-wrapper.h"
#include <math.h>
#include <chrono>

namespace py = pybind11;

struct Pair {
  Pair() {}

  Pair(int16_t f_, int16_t s_) {
    f = f_;
    s = s_;
  }

  int16_t f;
  int16_t s;
};


void get_best_path(py::array_t<int32_t> array, py::list &bestpath_lst, std::vector<int32_t> &texta,
                   std::vector<int32_t> &textb) {
  auto buf = array.request();
  int32_t *ptr = (int32_t *) buf.ptr;
  int32_t numr = array.shape()[0], numc = array.shape()[1];
  if (numr > 32000 || numc > 32000) throw std::runtime_error("Input array too large!");
  int16_t i = numr - 1, j = numc - 1;
  int32_t maxlen = numr + numc;
  std::queue<std::vector<Pair>> paths_to_explore;
  std::vector<Pair> bestpath;
  std::vector<Pair> path;
  int32_t best_continuous_match_len = -1;
  path.reserve(maxlen);
  path.push_back(Pair(i, j));
  int32_t max_paths_explore = 30000;
  int32_t paths_found = 0;
  while (true) {
    if (i == 0 && j == 0) {
      int32_t path_len = path.size();
      int32_t startidx = -1, endidx = -1;
      for (int32_t n = 1; n < path_len; n++) {
        Pair &pair = path[n];
        if (pair.f + 1 == path[n - 1].f && pair.s + 1 == path[n - 1].s) {
          if (startidx == -1) startidx = n;
          endidx = n;
        }
      }
      int32_t continuous_match_len = endidx - startidx;
//			std::cout << continuous_match_len <<  std::endl;
      if (bestpath.size() == 0 || continuous_match_len < best_continuous_match_len) {
        best_continuous_match_len = continuous_match_len;
        bestpath = path;
      }
      if (paths_to_explore.size() == 0) break;
      path = paths_to_explore.front();
      Pair &p = path.back();
      i = p.f, j = p.s;
      paths_to_explore.pop();
    }
    int32_t upc, leftc, diagc;
    int8_t idx = -1;
    if (i == 0) {
      idx = 1;
    } else if (j == 0) {
      idx = 0;
    } else {
      upc = ptr[(i - 1) * numc + j];
      leftc = ptr[i * numc + j - 1];
      diagc = ptr[(i - 1) * numc + j - 1];
    }
    if (idx != -1) { ;
    } else if (diagc < leftc && diagc < upc) {
      idx = 2;
    } else if (upc < leftc && upc != diagc && (texta[i] != textb[j] || upc + 1 < diagc)) {
      idx = 0;
    } else if (leftc < upc && leftc != diagc && (texta[i] != textb[j] || leftc + 1 < diagc)) {
      idx = 1;
    } else {

      if (leftc == diagc && upc == diagc) {
        idx = 2;
      } else if (leftc == upc) {
        if (leftc + 1 != diagc) {
          throw std::runtime_error("Should not be possible B");
        } else {
          if (paths_found < max_paths_explore) {
            std::vector<Pair> pathcopied(path);
            Pair explorep(i, j - 1);
            pathcopied.push_back(explorep);
            paths_to_explore.push(pathcopied);

            pathcopied = path;
            explorep = Pair(i - 1, j);
            pathcopied.push_back(explorep);
            paths_to_explore.push(pathcopied);
            paths_found++;
          }

          idx = 2;
        }
      } else if (leftc + 1 == diagc) {
        if (paths_found < max_paths_explore) {
          std::vector<Pair> pathcopied(path);
          Pair explorep(i, j - 1);
          pathcopied.push_back(explorep);
          paths_to_explore.emplace(pathcopied);
          paths_found++;
        }
        idx = 2;
      } else if (upc + 1 == diagc) {
        if (paths_found < max_paths_explore) {
          std::vector<Pair> pathcopied(path);
          Pair explorep(i - 1, j);
          pathcopied.push_back(explorep);
          paths_to_explore.emplace(pathcopied);
          paths_found++;
        }
        idx = 2;
      } else if (diagc <= upc && diagc <= leftc) {
        idx = 2;
      } else {
        throw std::runtime_error(
          "Should not be possible C " + std::to_string(leftc) + " " + std::to_string(upc) + " " +
          std::to_string(diagc));
      }
    }

    if (idx == 0) {
      i--;
    } else if (idx == 1) {
      j--;
    } else {
      i--, j--;
    }
    Pair newp = Pair(i, j);
    path.push_back(newp);
  }
  if (bestpath.size() == 1) throw std::runtime_error("No best path found!");
  for (int32_t k = 0; k < bestpath.size(); k++) {
    bestpath_lst.append(bestpath[k].f);
    bestpath_lst.append(bestpath[k].s);
  }
}


py::object calc_sum_cost(py::array_t<int32_t> array, std::vector<int32_t> &texta, std::vector<int32_t> &textb) {
  if (array.ndim() != 2)
    throw std::runtime_error("Input should be 2-D NumPy array");

  int M = array.shape()[0], N = array.shape()[1];
  if (M != texta.size() || N != textb.size()) throw std::runtime_error("Sizes do not match!");
  auto buf = array.request();
  int32_t *ptr = (int32_t *) buf.ptr;

  for (int32_t i = 0; i < M; i++) {
    for (int32_t j = 0; j < N; j++) {
      int32_t elem_cost = 2;
      if (texta[i] == textb[j]) elem_cost = 1;

      if (i == 0) {
        if (j == 0) {
          ptr[0] = 0;
        } else {
          ptr[j] = elem_cost + ptr[j - 1];
        }
      } else if (j == 0) {
        ptr[i * N] = elem_cost + ptr[(i - 1) * N];
      } else {
        int32_t upc = ptr[(i - 1) * N + j];
        int32_t leftc = ptr[i * N + j - 1];
        int32_t diagc = ptr[(i - 1) * N + j - 1];
        int32_t transition_cost = std::min(upc, std::min(leftc, diagc));
        if (diagc < leftc && diagc < upc) {
          if (elem_cost == 2) elem_cost = 4;
          transition_cost += elem_cost;
        } else {
          transition_cost += 2;
        }
        ptr[i * N + j] = transition_cost;
      }
    }
  }
  return py::cast<py::none>(Py_None);
}


class SegList {
public:
  std::vector<int> values;
  std::vector<int> end_indcs; // points at last value (not one past!)
  int iter_pointer;
  int seg_start_idx;
  int seg_end_idx;
  int seg_idx;

  SegList() {
    iter_pointer = 0;
    seg_idx = 0, seg_start_idx = 0, seg_end_idx = 0;
  }

  ~SegList() {}

  inline const int get_value(const int idx) const {
    return values.at(idx);
  }

  void AddElement(int state) {
    values.emplace_back(state);
  }

  void FinalizeList() {
    end_indcs.push_back(values.size() - 1);
  }

  int NumSegs() const {
    return end_indcs.size();
  }

  void IterSegs() {
    if (end_indcs.empty()) {
      return;
    }
    seg_end_idx = end_indcs[iter_pointer];
    seg_start_idx = iter_pointer == 0 ? 0 : end_indcs[iter_pointer - 1] + 1;
    seg_idx = seg_start_idx;
//    std::cout << "A "<<seg_start_idx<<" "<<seg_end_idx<<" "<<iter_pointer<<" "<<end_indcs.size()<<std::endl;
    iter_pointer++;
    if (iter_pointer == NumSegs()) {
      iter_pointer = 0;
    }
  }

  const int IterSeg() {
    seg_idx++;
    return get_value(seg_idx-1);
  }

  void AppendSubVec(int subvec_len) {
    if (end_indcs.empty()) throw "Should not be possible!";
    int start_idx = end_indcs.size() == 1 ? 0 : end_indcs[end_indcs.size() - 2] + 1;
    int size = values.size();
    for(int i = 0; i < subvec_len; i++) {
      int state = values[start_idx + i];
      AddElement(state);
    }
  }

  int CurrentSize() const {
    int start_idx = end_indcs.empty() ? 0 : end_indcs.back() + 1;
    return values.size() - start_idx;
  }

  std::vector< std::vector<int>> Simplify() {
    std::vector< std::vector<int>> alts;
    while (true) {
      this->IterSegs();
      std::vector<int> alt;
      while (this->seg_idx <= this->seg_end_idx) {
        alt.push_back(this->IterSeg());
      }
      alts.push_back(alt);
      if (this->iter_pointer == 0) break;
    }
    return alts;
  }
};


SegList* get_paths_fst(WrappedFst& ifst) {

  int start_state = ifst.GetStart();
  SegList* seglist = new SegList;
//  py::list paths;
  for (Arc& arc: ifst.GetArcs(start_state)) {
    std::vector<std::tuple<int, int, int>> arcs_to_visit;
    int nextstate = arc.nextstate, olabel = arc.olabel;
//    py::list path;
    while (true) {
      if (olabel != 0) {
        seglist->AddElement(olabel);
//        path.append(py::int_(olabel));
      }
      if (ifst.Final(nextstate) == 0.f) {
        if (arcs_to_visit.empty()) {
//          paths.append(path);
          seglist->FinalizeList();
          break;
        } else {
//          paths.append(path);
          seglist->FinalizeList();
          std::tuple<int, int, int> tpl = arcs_to_visit.back();
          arcs_to_visit.pop_back();
          int state = std::get<0>(tpl), j = std::get<1>(tpl), lst_len = std::get<2>(tpl);
//          py::slice slice(0, lst_len, 1);
//          path = path[slice];
          seglist->AppendSubVec(lst_len);
          int cnt = 0;
          for (Arc& c_arc: ifst.GetArcs(state)) {
            if (cnt == j) {
              nextstate = c_arc.nextstate, olabel = c_arc.olabel;
              break;
            }
            cnt++;
          }
        }
      } else {
        int state = nextstate;
        ArcIterator arc_iterator(ifst, state);
        if (arc_iterator.num_arcs != 0) {
          Arc c_arc = arc_iterator.Value();
          nextstate = c_arc.nextstate, olabel = c_arc.olabel;
          for (int i = 0; i < arc_iterator.num_arcs - 1; i++) {
            arcs_to_visit.emplace_back(std::make_tuple(state, i + 1, seglist->CurrentSize()));
//            arcs_to_visit.emplace_back(std::make_tuple(state, i + 1, path.size()));
          }
        }
      }
    }
  }
  return seglist;
}

py::int_ set_counts(SegList& sl, py::dict& subword_cnts, const py::int_& word_counts) {
  py::int_ total_word_count = 0;
  while (true) {
    sl.IterSegs();
    while (sl.seg_idx <= sl.seg_end_idx) {
      int subword = sl.IterSeg();
      if (subword < 0) {
        std::cout <<"Something is wrong!!!\n"<<std::endl;
        throw "Something is wrong!!!\n";
      }
      if (!subword_cnts.contains(subword)) {
        subword_cnts[py::int_(subword)] = py::int_(0);
      }
      py::int_ count = subword_cnts[py::int_(subword)];
      subword_cnts[py::int_(subword)] = count + word_counts;
      total_word_count += word_counts;
    }
    if (sl.iter_pointer == 0) break;
  }
  return total_word_count;
}

double calc_subword_prob(SegList& sl, py::dict& subword_prob_d, py::dict& subword_costs) {
  double word_likel_prob = 0.;
  while (true) {
    sl.IterSegs();
    double cost = 0.;
    std::vector<double> costs;
    while (sl.seg_idx <= sl.seg_end_idx) {
      int subword = sl.IterSeg();
      double subword_cost = subword_costs[py::int_(subword)].cast<double>();
      costs.push_back(subword_cost);
      cost += subword_cost;

    }

    sl.seg_idx = sl.seg_start_idx;
    int i = 0;
    while (sl.seg_idx <= sl.seg_end_idx) {
      int subword = sl.IterSeg();

      double value = exp(-(cost - costs[i]));
      py::int_ py_subword = py::int_(subword);
      double prev_value;
      if (subword_prob_d.contains(py_subword)) {
        subword_prob_d[py_subword] = 0.;
        prev_value = 0.;
      } else {
        prev_value = subword_prob_d[py::int_(subword)].cast<double>();
      }
      subword_prob_d[py_subword] = prev_value + value;
      i++;
    }

    double prob = exp(-cost);
    word_likel_prob += prob;
    if (sl.iter_pointer == 0) break;
  }
  return word_likel_prob;
}

double process_segs(SegList& sl, py::dict& subword_derivs, py::array_t<double>& subword_costs, std::vector<int> subwords,
                    double prob_sum) {
  double* costs_array = (double*) subword_costs.request().ptr;
  int sz = subword_costs.size();
  double hat_word_prob = 0.;
  std::vector<float> subword_to_factor(sz);
  for (int i = 0; i < sz; i++) {
    subword_to_factor[i] = 0.f;
  }
  std::unordered_set<int> subwords_used_set;
  while (true) {
    sl.IterSegs();
    while (sl.seg_idx <= sl.seg_end_idx) {
      int subword = sl.IterSeg();
      subwords_used_set.insert(subword);
    }
    if (sl.iter_pointer == 0) break;
  }
  std::vector<int> subwords_used;
  std::vector<int> subwords_unused;
  for (const int& subword: subwords) {
    if (subwords_used_set.find(subword) == subwords_used_set.end()) {
      subwords_unused.emplace_back(subword);
    } else {
      subwords_used.emplace_back(subword);
    }
  }

  int seg_count = 0;
  double const_factor = 0.;
  while (true) {
    sl.IterSegs();

    double subword_product = 0.;
    std::vector<int> alt_subwords;
    while (sl.seg_idx <= sl.seg_end_idx) {
      int subword = sl.IterSeg();
      alt_subwords.push_back(subword);
      subword_product += (-costs_array[subword] - log(prob_sum));
    }
    subword_product = exp(subword_product);
    hat_word_prob += subword_product;
    uint_least16_t seglen = (uint_least16_t) alt_subwords.size();

    for (int subword: alt_subwords) {
      py::int_ subword_py(subword);
      double subword_k_prob = exp(-costs_array[subword]) / prob_sum;
      double d_sum = 0.;
      for (int j = 0; j < seglen; j++) {
        int subword_j = alt_subwords[j];
        // x_i cancels out, no 1/x_i
        if (subword == subword_j) {
          d_sum += (1 - subword_k_prob);
        } else {
          d_sum += -subword_k_prob;
        }
      }
      subword_derivs[subword_py] = subword_derivs[subword_py].cast<double>() + subword_product * d_sum;
    }

    std::sort(alt_subwords.begin(), alt_subwords.end());
    int found_count = 0;
    const int* ptr = alt_subwords.data();
    double factor = seglen * subword_product;
    for (const int& subword: subwords_used) {
      if (found_count < seglen && ptr[found_count] == subword) {
        ++found_count;
        continue;
      }

      subword_to_factor[subword] += factor;
    }
    const_factor += factor;
//    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//    time_taken += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    if (sl.iter_pointer == 0) break;
    seg_count++;
  }

  for (const int& subword: subwords_unused) {
    subword_to_factor[subword] = const_factor;
  }

//  std::cout << "Time difference = " << time_taken << std::endl;
  for (int subword = 0; subword < subword_to_factor.size(); subword++) {
    float factor = subword_to_factor[subword];
    if (factor == 0.) continue;
    py::int_ subword_py(subword);
    double subword_k_prob = exp(-costs_array[subword]) / prob_sum;
    subword_derivs[subword_py] = subword_derivs[subword_py].cast<double>() - subword_k_prob * factor;
  }
  return hat_word_prob;
}

double process_words_and_segs(py::list& lst_alts, std::vector<double> lst_probs, py::array_t<double>& subword_derivs, py::array_t<double>& subword_costs, std::vector<int> subwords,
                              double prob_sum) {
  double total_likel = 0.;
  int count = lst_alts.size();
  double* costs_array = (double*) subword_costs.request().ptr;
  double* derivs_array = (double*) subword_derivs.request().ptr;
  int sz = subword_costs.size();
  std::vector<double> subword_to_factor(sz);
  std::vector<double> word_subword_derivs(sz);


//  SegList& sl_special = lst_alts[1275].cast<SegList&>();
//  std::string valss = "";
//  for (int j = 13785; j < 13795; j++) {
//    valss += std::to_string(sl_special.get_value(j)) + " ";
//  }
//  std::cout << valss<<std::endl;


  for (int i = 0; i < count; ++i) {
//    std::cout << "COUNT " << i<<std::endl;
    SegList& sl = lst_alts[i].cast<SegList&>();
//    std::cout << "val " << sl_special.get_value(13790)<<std::endl;
//    if (i == 1275) {
//      std::string s = "";
//      std::string is = "";
//      std::vector<int> idxs;
//      for (int j = 0; j < sl.values.size(); j++) {
//        if (sl.get_value(j) > 100000) {
//          idxs.push_back(j);
//          is += std::to_string(j) + " ";
//        }
//        s += std::to_string(sl.get_value(j)) + " ";
//      }
//      std::string vals = "";
//      for (int j = idxs[0] - 5; j < idxs[0] + 5; j++) {
//        vals += std::to_string(sl.get_value(j)) + " ";
//      }
//      std::cout << vals<<std::endl;
//      std::cout <<is<<std::endl;
//      std::cout << s << std::endl;
//    }
    std::cout << "max found "<< *std::max_element(sl.values.data(), sl.values.data() + sl.values.size()) << std::endl;
    double word_prob = lst_probs[i];

    double hat_word_prob = 0.;
    std::fill(subword_to_factor.begin(), subword_to_factor.end(), 0.);
    std::fill(word_subword_derivs.begin(), word_subword_derivs.end(), 0.);

    std::unordered_set<int> subwords_used_set;
    while (true) {
      sl.IterSegs();
      while (sl.seg_idx <= sl.seg_end_idx) {
        const int subword = sl.IterSeg();
        subwords_used_set.insert(subword);
      }
      if (sl.iter_pointer == 0) break;
    }
    std::cout <<"max "<< *std::max_element(subwords_used_set.cbegin(), subwords_used_set.cend()) << std::endl;
    std::vector<int> subwords_used;
    subwords_used.insert(subwords_used.end(), subwords_used_set.cbegin(), subwords_used_set.cend());
    std::sort(subwords_used.begin(), subwords_used.end());

    std::vector<int> subwords_unused;
    int check_idx = 0;
    for (const int& subword: subwords) {
      if (check_idx < subwords_used.size() && subword == subwords_used.at(check_idx)) {
        ++check_idx;
      } else {
        subwords_unused.emplace_back(subword);
      }
    }
    std::cout << "A"<<std::endl;
    int seg_count = 0;
    double const_factor = 0.;
    while (true) {
      sl.IterSegs();
      std::cout << "A1"<<std::endl;
      double subword_product = 0.;
      std::vector<int> alt_subwords;
      while (sl.seg_idx <= sl.seg_end_idx) {
        const int subword = sl.IterSeg();
        alt_subwords.push_back(subword);
        if (subword > sz) throw "\n\n\nERRRORORRR\n\n\n";
        subword_product += (-costs_array[subword] - log(prob_sum));
      }
      std::cout << "A2"<<std::endl;
      subword_product = exp(subword_product);
      hat_word_prob += subword_product;
      int seglen = alt_subwords.size();

      for (int subword: alt_subwords) {
        if (subword > sz) throw "\n\n\nERRRORORRR\n\n\n";
        double subword_k_prob = exp(-costs_array[subword]) / prob_sum;
        double d_sum = 0.;
        for (int j = 0; j < seglen; j++) {
          int subword_j = alt_subwords[j];
          // x_i cancels out, no 1/x_i
          if (subword == subword_j) {
            d_sum += (1 - subword_k_prob);
          } else {
            d_sum += -subword_k_prob;
          }
        }
        word_subword_derivs.at(subword) += subword_product * d_sum;
      }
      std::cout << "A3 "<<seglen<<std::endl;
      std::sort(alt_subwords.begin(), alt_subwords.end());
      int found_count = 0;
      const int* ptr = alt_subwords.data();
      double factor = seglen * subword_product;
      std::cout << *std::max_element(subwords_used.cbegin(), subwords_used.cend()) <<" "<<subword_to_factor.size()<< std::endl;
      for (const int& subword: subwords_used) {
//        std::cout << found_count << std::endl;
        if (found_count < seglen && ptr[found_count] == subword) {
          ++found_count;
          continue;
        }
        if (subword > 30000) std::cout << "YOWTF ??" <<std::endl;
        subword_to_factor.at(subword) += factor;
      }
      std::cout << "A4" <<std::endl;
      const_factor += factor;
      if (sl.iter_pointer == 0) break;
      seg_count++;
    }
    std::cout << "B"<<std::endl;
    for (const int& subword: subwords_unused) {
      subword_to_factor.at(subword) = const_factor;
    }

//  std::cout << "Time difference = " << time_taken << std::endl;
    for (int subword = 0; subword < subword_to_factor.size(); subword++) {
      float factor = subword_to_factor.at(subword);
      if (factor == 0.) continue;
      if (subword > sz) throw "\n\n\nERRRORORRR\n\n\n";
      double subword_k_prob = exp(-costs_array[subword]) / prob_sum;
      word_subword_derivs.at(subword) -= subword_k_prob * factor;
    }
    std::cout << "C"<<std::endl;
    double ratio = word_prob / hat_word_prob;
    for (int subword: subwords) {
      if (subword > sz) throw "\n\n\nERRRORORRR\n\n\n";
      derivs_array[subword] += word_subword_derivs.at(subword) * ratio;
    }
    total_likel += word_prob * log(hat_word_prob);
    std::cout << "D"<<std::endl;
  }
  return total_likel;
}

SegList* filter_segs(SegList& sl, std::vector<int> letters_aligned, std::vector< std::vector<int>> phones_aligned, py::array_t<int64_t>& osyms_len) {
  int64_t* osym_len_ptr = (int64_t*) osyms_len.request().ptr;
  SegList* new_sl = new SegList;
  if (!sl.NumSegs()) return new_sl;
  while (true) {
    sl.IterSegs();
    bool keep_seg = true;
    int i = 0;
//    std::cout << "A"<<std::endl;
    while (sl.seg_idx <= sl.seg_end_idx - 1) {  // skip last because checking boundaries
      int subword = sl.IterSeg();
      int subword_len = osym_len_ptr[subword];
      std::vector<int> phones;
      int end_idx = i + subword_len;
//      std::cout << i<<" "<<end_idx<<std::endl;
      for (; i < end_idx; i++) {
        for (int j = 0; j < phones_aligned[i].size(); j++) {
          phones.push_back(phones_aligned[i][j]);
        }
      }

      if ((subword_len == 1 && (rand() % 4) == 0) ||
           phones.empty()) {
        keep_seg = false;
      }
    }

    if (keep_seg) {
      sl.seg_idx = sl.seg_start_idx;
      while (sl.seg_idx <= sl.seg_end_idx) {
        int subword = sl.IterSeg();
        new_sl->AddElement(subword);
      }
      new_sl->FinalizeList();
    }
    if (sl.iter_pointer == 0) break;
  }
//  std::cout << "Done"<<std::endl;
  return new_sl;
}


PYBIND11_MODULE(fast, m) {
  m.doc() = "pybind11 plugin";
  m.def("calc_sum_cost", &calc_sum_cost, "Calculate summed cost matrix");
  m.def("get_best_path", &get_best_path, "get_best_path");
  m.def("get_paths_fst", &get_paths_fst, py::return_value_policy::take_ownership);
  m.def("set_counts", &set_counts);
  m.def("calc_subword_prob", &calc_subword_prob);
  m.def("process_segs", &process_segs);
  m.def("process_words_and_segs", &process_words_and_segs);
  m.def("filter_segs", &filter_segs, py::return_value_policy::take_ownership);

  py::class_<Arc>(m, "Arc")
    .def(py::init<int, int, double, int>())
    .def_readwrite("ilabel", &Arc::ilabel)
    .def_readwrite("olabel", &Arc::olabel)
    .def_readwrite("weight", &Arc::weight)
    .def_readwrite("nextstate", &Arc::nextstate);

  py::class_<WrappedFst>(m, "WrappedFst")
    .def(py::init<>())
    .def("read", &WrappedFst::Read)
    .def("write", &WrappedFst::Write)
    .def("write_ark_entry", &WrappedFst::WriteArkEntry)
    .def("read_ark_entries", &WrappedFst::ReadArkEntries)
    .def("add_state", &WrappedFst::AddState)
    .def("set_start", &WrappedFst::SetStart)
    .def("set_final", &WrappedFst::SetFinal, py::arg("state"),py::arg("weight")=0.)
    .def("add_arc", &WrappedFst::AddArc)
    .def("get_start", &WrappedFst::GetStart)
    .def("get_arcs", &WrappedFst::GetArcs)
    .def("determinize", &WrappedFst::Determinize)
    .def("minimize", &WrappedFst::Minimize)
    .def("arc_sort", &WrappedFst::ArcSort)
    .def("compose", &WrappedFst::Compose)
    .def("shortest_path", &WrappedFst::ShortestPath)
    .def("final", &WrappedFst::Final)
    .def("is_final", &WrappedFst::isFinal)
    .def("states", &WrappedFst::States)
    .def("connect", &WrappedFst::Connect)
    .def("delete_arcs", &WrappedFst::DeleteArcs)
    .def("delete_states", &WrappedFst::DeleteStates)
    .def("num_states", &WrappedFst::NumStates)
    .def("num_arcs", &WrappedFst::NumArcs)
    .def("insert", &WrappedFst::Insert)
    .def("replace_single", &WrappedFst::ReplaceSingle)
    .def("add_boost", &WrappedFst::AddBoost)
    .def("normalise_weights", &WrappedFst::NormaliseWeights)
    .def("copy", &WrappedFst::Copy,  py::return_value_policy::take_ownership)
    .def(py::pickle(
      [](const WrappedFst& f) {
        int num_states = f.NumStates();
        int start_state = f.GetStart();
        py::list final_states;
        py::list arcs;
        for (int state: f.States()) {
          if (f.Final(state) == 0.) {
            final_states.append(py::int_(state));
          }
          for (Arc& arc: f.GetArcs(state)) {
            arcs.append(py::make_tuple(py::int_(state), py::int_(arc.nextstate), py::int_(arc.ilabel), py::int_(arc.olabel), py::float_(arc.weight)));
          }
        }
        return py::make_tuple(num_states, start_state, final_states, arcs);
      },
      [](py::tuple t) {
        WrappedFst f;
        int num_states = t[0].cast<int>();
        int start_state = t[1].cast<int>();
        py::list final_states = t[2];
        py::list arcs = t[3];
        for (int i = 0; i < num_states; i++) {
          int state = f.AddState();
          if (state == start_state) f.SetStart(state);
        }
        for (py::handle obj: final_states) {
          int state = obj.cast<int>();
          f.SetFinal(state);
        }
        for (py::handle obj: arcs) {
          py::tuple tpl = obj.cast<py::tuple>();
          f.AddArc(tpl[0].cast<int>(), tpl[1].cast<int>(), tpl[2].cast<int>(), tpl[3].cast<int>(), tpl[4].cast<double>());
        }
        return f;
      }
      ))
      .def("__copy__", [](const WrappedFst& wfst) {
        return WrappedFst(wfst);
      })
      .def("__deepcopy__", [](const WrappedFst& wfst) {
        return WrappedFst(wfst);
      });

  py::class_<ArcIterator>(m, "ArcIterator")
    .def(py::init<WrappedFst&, int>())
    .def("Done", &ArcIterator::Done)
    .def("Next", &ArcIterator::Next)
    .def("Value", &ArcIterator::Value)
    .def("SetValue", &ArcIterator::SetValue);

  py::class_<SegList>(m, "SegList")
    .def(py::init<>())
    .def_readwrite("iter_pointer", &SegList::iter_pointer)
    .def_readwrite("seg_idx", &SegList::seg_idx)
    .def_readwrite("seg_end_idx", &SegList::seg_end_idx)
    .def_readwrite("end_indcs", &SegList::end_indcs)
    .def_readwrite("values", &SegList::values)
    .def("AddElement", &SegList::AddElement)
    .def("FinalizeList", &SegList::FinalizeList)
    .def("NumSegs", &SegList::NumSegs)
    .def("IterSegs", &SegList::IterSegs)
    .def("IterSeg", &SegList::IterSeg, py::return_value_policy::move)
    .def("Simplify",&SegList::Simplify)
    .def(py::pickle(
      [](const SegList& l) {
        py::array_t<int> values({l.values.size()}, {sizeof(int)}, l.values.data());
        py::array_t<int> indcs({l.end_indcs.size()}, {sizeof(int)}, l.end_indcs.data());
        return py::make_tuple(values, indcs);
      },
      [](const py::tuple& t) {
        SegList* l = new SegList();
        py::array_t<int> values = t[0].cast<py::array_t<int>>();
        py::array_t<int> indcs = t[1].cast<py::array_t<int>>();
        l->values.resize(values.size());
        l->end_indcs.resize(indcs.size());
        memcpy(l->values.data(), (int*) values.request().ptr, values.size() * sizeof(int));
        memcpy(l->end_indcs.data(), (int*) indcs.request().ptr, indcs.size() * sizeof(int));
        return l;
      }
      ));

}