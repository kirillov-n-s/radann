#pragma once
#include <vector>
#include <list>
#include "backward.h"

namespace radann::diff
{
    template<typename T>
    class tape_context;

    template<typename T>
    class tape
    {
    private:
        struct statement
        {
            index_t gradient;
            size_t last_term;
        };

        struct term
        {
            index_t gradient;
            array_no_ad<T> multiplier;
            backward_function<T> backward;
        };

        struct gradient
        {
            array_no_ad<T> value;
            size_t nrefs = 1;
        };

        std::vector<statement> _statements;
        std::vector<term> _terms;
        std::vector<gradient> _gradients;

        std::list<index_t> _gaps;
        index_t _next_index = 0;

        size_t _calls = 0;
        size_t _matches = 0;

        tape() = default;

    public:
        tape(const tape&) = delete;
        friend class tape_context<T>;

        index_t create_grad(const core::shape&);
        index_t derive_grad(index_t, const core::shape&, size_t);
        index_t copy_grad(index_t);

        void delete_grad(index_t);

        array_no_ad<T> get_grad(index_t) const;

        template<typename Tag, typename Expr>
        void push_term(index_t, const expr::base<Expr>&);
        void push_statement(index_t);

        void reverse();
        void clear();

        void stats() const
        {
            std::cout << "Gradient indices: " << _next_index << '\n'
                      << "Shape matches: " << _matches << "/" << _calls << '\n';
        }
    };
}

namespace radann::diff
{
    template<typename T>
    index_t tape<T>::create_grad(const core::shape &shape)
    {
        _calls++;
        if (_gaps.empty())
        {
            _gradients.push_back({ array_no_ad<T> { shape } });
            return _next_index++;
        }

        auto index = _gaps.front();
        _gaps.pop_front();
        auto& gap = _gradients[index];
        gap.nrefs = 1;
        auto& grad = gap.value;
        if (grad.shape() == shape)
        {
            _matches++;
            grad.zero();
        }
        else
            grad >>= array_no_ad<T> { shape };
        return index;
    }

    template<typename T>
    index_t tape<T>::derive_grad(index_t base_index, const core::shape &shape, size_t offset)
    {
        array_no_ad<T> grad {_gradients[base_index].value.storage(), shape, offset, base_index };
        if (_gaps.empty())
        {
            _gradients.push_back({ grad });
            return _next_index++;
        }

        auto index = _gaps.front();
        _gaps.pop_front();
        auto& gap = _gradients[index];
        gap.nrefs = 1;
        gap.value >>= grad;
        return index;
    }

    template<typename T>
    index_t tape<T>::copy_grad(index_t index)
    {
        _gradients[index].nrefs++;
        return index;
    }

    template<typename T>
    void tape<T>::delete_grad(index_t index)
    {
        _gradients[index].nrefs--;
    }

    template<typename T>
    array_no_ad<T> tape<T>::get_grad(index_t index) const
    {
        return _gradients[index].value;
    }

    template<typename T>
    template<typename Tag, typename Expr>
    void tape<T>::push_term(index_t index, const expr::base<Expr> &mult)
    {
        _terms.push_back({ index, array_no_ad<T> { mult }, &backward<Tag>::function });
    }

    template<typename T>
    void tape<T>::push_statement(index_t index)
    {
        _statements.push_back({ index, _terms.size() });
    }

    template<typename T>
    void tape<T>::reverse()
    {
        for (auto i = _statements.size() - 1; i > 0; i--)
        {
            const auto& statement = _statements[i];
            auto& output_grad = _gradients[statement.gradient].value;
            auto output_grad_copy = core::copy(output_grad);
            output_grad.zero();
            for (auto j = _statements[i - 1].last_term; j < statement.last_term; j++)
            {
                const auto& term = _terms[j];
                term.backward(_gradients[term.gradient].value, output_grad_copy, term.multiplier);
            }
        }

        const auto& statement = _statements[0];
        auto& output_grad = _gradients[statement.gradient].value;
        auto output_grad_copy = core::copy(output_grad);
        output_grad.zero();
        for (auto j = 0; j < statement.last_term; j++)
        {
            const auto& term = _terms[j];
            term.backward(_gradients[term.gradient].value, output_grad_copy, term.multiplier);
        }
    }

    template<typename T>
    void tape<T>::clear()
    {
        _statements.clear();
        _terms.clear();
        auto n = _gradients.size();
        for (auto i = 0; i < n; i++)
            if (_gradients[i].nrefs == 0)
                _gaps.push_back(i);
    }
}
